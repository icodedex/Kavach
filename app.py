


import os
import asyncio
import json
import random
from typing import Set
from loguru import logger

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# DB imports (async SQLAlchemy 1.4 compatible)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, func, Text

# MQTT
from asyncio_mqtt import Client as MQTTClient, MqttError

# Load .env if present
from dotenv import load_dotenv
load_dotenv()

# --- CONFIG ---
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "kavach/sensors/#")
TCP_HOST = os.getenv("TCP_HOST", "0.0.0.0")
TCP_PORT = int(os.getenv("TCP_PORT", 9000))
DB_URL = os.getenv("DB_URL", "sqlite+aiosqlite:///./kavach.db")
MODEL_PATH = os.getenv("MODEL_PATH", None)
WS_PATH = os.getenv("WS_BROADCAST_PATH", "/ws/alerts")

# --- DATABASE SETUP ---
Base = declarative_base()
engine = create_async_engine(DB_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class SensorEvent(Base):
    __tablename__ = "sensor_events"
    id = Column(Integer, primary_key=True, index=True)
    sensor_type = Column(String, nullable=False)
    source = Column(String, nullable=False)
    raw_payload = Column(Text, nullable=False)
    parsed_value = Column(Text)
    prediction = Column(Text)
    risk_level = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# --- PREPROCESSING (basic) ---
def parse_sensor_message(sensor_type: str, raw: bytes) -> dict:
    """Minimal parser: expects JSON bytes for demo sensors."""
    try:
        data = json.loads(raw.decode(errors="ignore"))
    except Exception:
        return {"raw": raw.decode(errors="ignore"), "error": "invalid_json"}
    if sensor_type == "lidar":
        return {"distance": float(data.get("distance", -1)), "confidence": float(data.get("c", 0))}
    if sensor_type == "radar":
        return {"speed": float(data.get("speed", 0)), "range": float(data.get("range", -1))}
    if sensor_type == "camera":
        # camera might send visibility or metadata
        return {"visibility": float(data.get("visibility", 1.0)), "frame_id": data.get("id")}
    return data

# --- MODEL STUB ---
# put at top: import torch
import torch
import torchvision.transforms as T
from PIL import Image
import base64
import io

class PyTorchModelRunner:
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("No model_path provided; using dummy heuristic.")

    def load_model(self, path):
        # example: model saved with torch.jit.save or torch.save(state_dict)
        try:
            # if you have a scripted model:
            self.model = torch.jit.load(path, map_location=self.device)
            self.model.eval()
            logger.info(f"Loaded scripted model from {path} to {self.device}")
        except Exception:
            # try state_dict style (requires model class)
            logger.exception("Failed to load scripted model; implement state_dict loader for your architecture.")

    async def predict(self, parsed_payload: dict) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_predict, parsed_payload)

    def _sync_predict(self, parsed_payload: dict) -> dict:
        # Example: if camera sends base64 frame named 'frame_b64'
        visibility = parsed_payload.get("visibility")
        if visibility is not None:
            # you might directly return this if your model uses it
            v = float(visibility)
            risk = "HIGH" if v < 0.3 else ("MEDIUM" if v < 0.6 else "LOW")
            return {"fog_intensity": v, "risk": risk}

        # If we have image data as base64:
        frame_b64 = parsed_payload.get("frame_b64")
        if frame_b64 and self.model:
            try:
                img_bytes = base64.b64decode(frame_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
                x = transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(x)  # adapt to your model's output
                    fog_score = float(torch.sigmoid(out).item()) if isinstance(out, torch.Tensor) else float(out)
                    risk = "HIGH" if fog_score < 0.3 else ("MEDIUM" if fog_score < 0.6 else "LOW")
                    return {"fog_intensity": fog_score, "risk": risk}
            except Exception:
                logger.exception("Model inference failed; falling back to heuristic.")
        # fallback heuristic
        v = random.random()
        return {"fog_intensity": v, "risk": "HIGH" if v < 0.3 else ("MEDIUM" if v < 0.6 else "LOW")}

# --- APPLICATION ---
app = FastAPI(title="Kavach Backend (single-file)")

# Shared queue for inter-task communication
message_queue: asyncio.Queue = asyncio.Queue()
model_runner = ModelRunner(MODEL_PATH)

# WebSocket connection set
connected_websockets: Set[WebSocket] = set()

# --- MQTT BACKGROUND LOOP ---
async def mqtt_loop(queue: asyncio.Queue):
    logger.info(f"Starting MQTT client -> broker {MQTT_BROKER}:{MQTT_PORT}, topic={MQTT_TOPIC}")
    try:
        async with MQTTClient(MQTT_BROKER, port=MQTT_PORT) as client:
            async with client.unfiltered_messages() as messages:
                await client.subscribe(MQTT_TOPIC)
                logger.info(f"Subscribed to MQTT {MQTT_TOPIC}")
                async for msg in messages:
                    topic = msg.topic
                    payload = msg.payload  # bytes
                    # push a tuple to queue: ("mqtt", topic, payload)
                    await queue.put(("mqtt", topic, payload))
    except MqttError as me:
        logger.error(f"MQTT client error: {me}")
        # retry after a short delay
        await asyncio.sleep(5)
        asyncio.create_task(mqtt_loop(queue))

# --- TCP SOCKET SERVER (hardware) ---
async def handle_tcp_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    logger.info(f"TCP connection from {addr}")
    try:
        while True:
            data = await reader.readline()
            if not data:
                break
            # put to queue as ("tcp", address_str, data)
            await message_queue.put(("tcp", str(addr), data))
    except Exception as e:
        logger.exception("TCP handler exception")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        logger.info(f"TCP connection closed: {addr}")

async def start_tcp_server(host: str = TCP_HOST, port: int = TCP_PORT):
    server = await asyncio.start_server(handle_tcp_connection, host, port)
    logger.info(f"TCP server listening on {host}:{port}")
    async with server:
        await server.serve_forever()

# --- PROCESSOR: consume queue, preprocess, predict, DB save, broadcast ---
async def broadcast_alert(alert: dict):
    living = set()
    for ws in list(connected_websockets):
        try:
            await ws.send_json(alert)
            living.add(ws)
        except Exception:
            logger.info("Failed to send to a WS client; removing")
    connected_websockets.clear()
    connected_websockets.update(living)

async def process_messages_loop():
    logger.info("Message processor started.")
    while True:
        try:
            source, topic_or_addr, payload = await message_queue.get()
            # derive sensor_type
            sensor_type = "unknown"
            if source == "mqtt":
                # topic like kavach/sensors/<type> or kavach/sensors/<type>/id
                try:
                    sensor_type = topic_or_addr.split("/")[-1]
                except Exception:
                    sensor_type = "mqtt"
            elif source == "tcp":
                # simple attempt: assume the payload is JSON with a 'type' field
                try:
                    j = json.loads(payload.decode(errors="ignore"))
                    sensor_type = j.get("type", "tcp")
                except Exception:
                    sensor_type = "tcp"

            parsed = parse_sensor_message(sensor_type, payload)
            prediction = await model_runner.predict(parsed)

            alert = {
                "sensor": sensor_type,
                "source": source,
                "parsed": parsed,
                "prediction": prediction,
            }

            # broadcast alert to all connected websockets (non-blocking)
            await broadcast_alert(alert)

            # persist to DB (async)
            try:
                async with AsyncSessionLocal() as session:
                    ev = SensorEvent(
                        sensor_type=sensor_type,
                        source=source,
                        raw_payload=payload.decode(errors="ignore"),
                        parsed_value=json.dumps(parsed),
                        prediction=json.dumps(prediction),
                        risk_level=prediction.get("risk", "UNKNOWN"),
                    )
                    session.add(ev)
                    await session.commit()
            except Exception:
                logger.exception("DB write failed")
        except Exception:
            logger.exception("Processor loop error")
            await asyncio.sleep(0.5)

# --- FASTAPI endpoints & WS endpoint ---
@app.on_event("startup")
async def on_startup():
    logger.info("Starting Kavach backend (startup).")
    # create tables if not exist
    await create_tables()
    # start background tasks
    loop = asyncio.get_event_loop()
    loop.create_task(mqtt_loop(message_queue))
    loop.create_task(start_tcp_server(TCP_HOST, TCP_PORT))
    loop.create_task(process_messages_loop())

@app.get("/api/ping")
async def ping():
    return JSONResponse({"status": "ok"})

@app.websocket(WS_PATH)
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    connected_websockets.add(ws)
    logger.info("WebSocket client connected.")
    try:
        while True:
            # keep connection alive; handle pings or client messages (optional)
            msg = await ws.receive_text()
            logger.debug(f"WS message from client: {msg}")
    except WebSocketDisconnect:
        connected_websockets.discard(ws)
        logger.info("WebSocket client disconnected.")
    except Exception:
        connected_websockets.discard(ws)
        logger.exception("WebSocket error")

# --- Simple sensor POST endpoint (helpful for testing without MQTT) ---
@app.post("/api/sensor/{sensor_type}")
async def ingest_sensor(sensor_type: str, payload: dict):
    """
    Accepts JSON POSTs for quick testing:
    POST /api/sensor/lidar  { "distance": 12.3, "c": 0.98 }
    """
    raw = json.dumps(payload).encode()
    await message_queue.put(("http", sensor_type, raw))
    return {"status": "queued", "sensor": sensor_type}

from fastapi import Query
from sqlalchemy import select, desc

@app.get("/api/events")
async def get_events(sensor: str | None = None, risk: str | None = None, limit: int = 50, offset: int = 0):
    async with AsyncSessionLocal() as session:
        q = select(SensorEvent)
        if sensor:
            q = q.where(SensorEvent.sensor_type == sensor)
        if risk:
            q = q.where(SensorEvent.risk_level == risk)
        q = q.order_by(desc(SensorEvent.timestamp)).limit(limit).offset(offset)
        res = await session.execute(q)
        rows = res.scalars().all()
        return {"count": len(rows), "events": [ {
            "id": r.id, "sensor": r.sensor_type, "source": r.source,
            "parsed": json.loads(r.parsed_value) if r.parsed_value else None,
            "prediction": json.loads(r.prediction) if r.prediction else None,
            "risk": r.risk_level, "timestamp": r.timestamp.isoformat()
        } for r in rows ]}


# --- Run with uvicorn when executed directly ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Running app directly using uvicorn.")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

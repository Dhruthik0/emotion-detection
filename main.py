# main.py
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from deepface import DeepFace

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# --------------- App setup ---------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------- DB setup ---------------
# Provide DATABASE_URL env var for online Postgres (e.g. postgres://user:pass@host:5432/db)
# If not set, fallback to a local sqlite file: sqlite:///./data/emotions.db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/emotions.db")

# Ensure local data folder exists when using sqlite
if DATABASE_URL.startswith("sqlite") and not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)

# Create SQLAlchemy engine
engine: Engine = create_engine(DATABASE_URL, echo=False, future=True)


def init_db():
    """Create the records table. Use Postgres SERIAL for non-sqlite, and integer primary key for sqlite."""
    try:
        if DATABASE_URL.startswith("sqlite"):
            ddl = """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                dominant_emotion TEXT,
                emotions_json TEXT,
                region_json TEXT,
                created_at TEXT
            )
            """
        else:
            # Postgres-friendly DDL
            ddl = """
            CREATE TABLE IF NOT EXISTS records (
                id SERIAL PRIMARY KEY,
                name TEXT,
                dominant_emotion TEXT,
                emotions_json TEXT,
                region_json TEXT,
                created_at TIMESTAMP
            )
            """
        with engine.begin() as conn:
            conn.execute(text(ddl))
    except SQLAlchemyError as e:
        print("DB init error:", e)


init_db()

# --------------- helpers ---------------
def read_imagefile_to_cv2(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# --------------- Routes ---------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image blob, runs DeepFace emotion analysis, returns:
    { dominant_emotion, emotions, region }
    """
    try:
        contents = await file.read()
        img = read_imagefile_to_cv2(contents)
        if img is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)

        # Convert BGR->RGB for DeepFace
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run DeepFace emotion analysis (enforce_detection=False avoids hard failure when no face)
        result = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        emotion = result.get("dominant_emotion", "")
        emotions = result.get("emotion", {}) or {}
        region = result.get("region", {}) or {}
        region = {k: int(region.get(k, 0) or 0) for k in ("x", "y", "w", "h")}

        response: Dict[str, Any] = {
            "dominant_emotion": emotion,
            "emotions": emotions,
            "region": region,
        }
        return JSONResponse(response)

    except Exception as e:
        # Log error server-side for debugging
        print("Error in /predict:", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/save")
async def save_record(
    name: str = Form(...),
    dominant_emotion: str = Form(...),
    emotions_json: str = Form(...),
    region_json: str = Form(...)
):
    """
    Save a detection to DB. Server creates timestamp (UTC).
    """
    try:
        created_at = datetime.utcnow().isoformat()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO records (name, dominant_emotion, emotions_json, region_json, created_at) "
                    "VALUES (:name, :dominant_emotion, :emotions_json, :region_json, :created_at)"
                ),
                {
                    "name": name,
                    "dominant_emotion": dominant_emotion,
                    "emotions_json": emotions_json,
                    "region_json": region_json,
                    "created_at": created_at,
                },
            )
        return JSONResponse({"status": "ok", "created_at": created_at})
    except Exception as e:
        print("Error saving record:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/records")
def list_records(name: Optional[str] = Query(None), limit: int = 100):
    """
    If name provided, return records for that name (recent first).
    Otherwise return recent records overall.
    """
    try:
        if name:
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT id, name, dominant_emotion, emotions_json, region_json, created_at "
                        "FROM records WHERE name = :name ORDER BY id DESC LIMIT :limit"
                    ),
                    {"name": name, "limit": limit},
                )
                raw_rows = result.mappings().all()  # list of RowMapping
        else:
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT id, name, dominant_emotion, emotions_json, region_json, created_at "
                        "FROM records ORDER BY id DESC LIMIT :limit"
                    ),
                    {"limit": limit},
                )
                raw_rows = result.mappings().all()

        # Convert RowMapping -> plain dict so we can modify fields
        rows = [dict(r) for r in raw_rows]

        # Parse JSON fields into Python objects and remove raw JSON columns
        for r in rows:
            ej = r.get("emotions_json")
            if isinstance(ej, str):
                try:
                    r["emotions"] = json.loads(ej)
                except Exception:
                    r["emotions"] = {}
            else:
                r["emotions"] = ej or {}

            rj = r.get("region_json")
            if isinstance(rj, str):
                try:
                    r["region"] = json.loads(rj)
                except Exception:
                    r["region"] = {}
            else:
                r["region"] = rj or {}

            # remove the raw JSON columns
            r.pop("emotions_json", None)
            r.pop("region_json", None)

        return JSONResponse({"records": rows})
    except Exception as e:
        print("Error reading records:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/healthz")
def health():
    return {"status": "ok"}

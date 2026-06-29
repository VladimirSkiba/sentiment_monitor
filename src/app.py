

import os
import sqlite3
import time
from datetime import datetime, timezone
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.joblib")
DB_PATH = os.path.join(BASE_DIR, "logs", "predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app = FastAPI(
    title="Sentiment Mini-Service",
    description='Демо ML-сервис: классификация тональности отзывов + логирование для мониторинга',
    version="1.0.0",
)

_model = None



def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Модель не найдена по пути {MODEL_PATH}. Сначала запустите src/train.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            request_id TEXT,
            text TEXT,
            predicted_label INTEGER,
            confidence REAL,
            latency_ms REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст отзыва для анализа")
    request_id: str | None = Field(
        default=None, description="Опциональный внешний id (например, id отзыва из стрима)"
    )


class PredictResponse(BaseModel):
    request_id: str
    predicted_label: int
    label_name: str
    confidence: float
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    model = get_model()
    start = time.perf_counter()
    proba = model.predict_proba([req.text])[0]
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    predicted_label = int(proba.argmax())
    confidence = float(proba.max())
    request_id = req.request_id or f"req_{int(time.time() * 1000)}"

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        (
            request_id,
            req.text,
            predicted_label,
            confidence,
            latency_ms,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return PredictResponse(
        request_id=request_id,
        predicted_label=predicted_label,
        label_name="positive" if predicted_label == 1 else "negative",
        confidence=round(confidence, 4),
        latency_ms=latency_ms,
    )



@app.get("/predict/history")
def history(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT request_id, text, predicted_label, confidence, latency_ms, created_at "
        "FROM predictions ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    cols = ["request_id", "text", "predicted_label", "confidence", "latency_ms", "created_at"]
    return [dict(zip(cols, row)) for row in rows]

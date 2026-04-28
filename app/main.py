from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path

from app.schemas import PredictRequest, PredictResponse
from app.predictor import load_all_models, predict, models_loaded

# ── Lifespan: load models on startup ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up - loading models...")
    load_all_models()
    print("Ready.")
    yield
    print("Shutting down.")

# ── App ──
app = FastAPI(
    title="Fake Review Detector",
    description="Bilingual fake review detection - English (BERT) and Arabic (CamelBERT)",
    version="1.0.0",
    lifespan=lifespan
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Routes ──
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": models_loaded()}

@app.post("/predict", response_model=PredictResponse)
def predict_review(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be empty.")
    if len(text) > 5000:
        raise HTTPException(status_code=422, detail="Text too long. Max 5000 characters.")
    result = predict(text)
    return PredictResponse(**result)
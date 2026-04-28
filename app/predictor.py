import re,  os
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ──

EN_REPO = "raed-7/fake-review-distilbert-en"
AR_REPO = "raed-7/fake-review-camelbert-ar"

HF_TOKEN = os.getenv("HF_TOKEN")

# ── Language detection ──
ARABIC_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')

def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "english"
    arabic_chars = len(ARABIC_RE.findall(text))
    ratio = arabic_chars / max(len(text.strip()), 1)
    return "arabic" if ratio > 0.15 else "english"

# ── Model storage ──
_models = {}

def load_all_models():
    """Load all models once at startup."""
    print("Loading English DistilBERT...")
    _models["en_tokenizer"] = AutoTokenizer.from_pretrained(EN_REPO, token=HF_TOKEN)
    _models["en_model"]     = AutoModelForSequenceClassification.from_pretrained(EN_REPO, token=HF_TOKEN)
    _models["en_model"].eval()
    _models["en_model"].to("cpu")
    print("English DistilBERT loaded.")

    print("Loading Arabic CamelBERT...")
    _models["ar_tokenizer"] = AutoTokenizer.from_pretrained(AR_REPO, token=HF_TOKEN)
    _models["ar_model"]     = AutoModelForSequenceClassification.from_pretrained(AR_REPO, token=HF_TOKEN)
    _models["ar_model"].eval()
    _models["ar_model"].to("cpu")
    print("Arabic CamelBERT loaded.")

    print("Skipping classical models (not deployed).")
    print("All models loaded successfully.")

def models_loaded() -> bool:
    return len(_models) > 0

# ── Inference ──
def predict(text: str) -> dict:
    start = time.time()
    lang  = detect_language(text)

    if lang == "arabic":
        result = _predict_arabic(text)
    else:
        result = _predict_english(text)

    result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    result["language"]           = lang
    return result

def _predict_english(text: str) -> dict:
    tokenizer = _models["en_tokenizer"]
    model     = _models["en_model"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    # Removing the token_type_ids — DistilBERT does not use them
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = F.softmax(logits, dim=-1)[0]
    label_idx  = probs.argmax().item()
    confidence = round(probs[label_idx].item() * 100, 1)
    label      = "Fake" if label_idx == 1 else "Real"

    return {
        "label":      label,
        "confidence": confidence,
        "model_used": "DistilBERT (distilbert-base-uncased)"
    }

def _predict_arabic(text: str) -> dict:
    tokenizer = _models["ar_tokenizer"]
    model     = _models["ar_model"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = F.softmax(logits, dim=-1)[0]
    label_idx  = probs.argmax().item()
    confidence = round(probs[label_idx].item() * 100, 1)
    label      = "Fake" if label_idx == 1 else "Real"

    return {
        "label":      label,
        "confidence": confidence,
        "model_used": "CamelBERT (camelbert-mix)"
    }
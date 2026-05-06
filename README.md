<div align="center">

# 🕵️ Bilingual Fake Review Detector

### English & Arabic · Real-time Detection · Powered by Transformers

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Online-success?style=for-the-badge)](https://fake-review-detector-eajs.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Models-yellow?style=for-the-badge)](https://huggingface.co/raed-7)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

**A FastAPI web application that classifies product reviews as `Fake` or `Real` in both English and Arabic, using fine-tuned transformer models.**

[🌐 **Try the Live App**](https://fake-review-detector-eajs.onrender.com/) · [📖 Documentation](#-api) · [🚀 Quick Start](#-run-locally)

</div>

---

## ✨ Features

- 🌍 **Bilingual support** — automatic English / Arabic routing via Unicode-script heuristic
- 🎯 **High accuracy** — DistilBERT for English, CamelBERT-Mix for Arabic
- ⚡ **Real-time inference** — models loaded once at startup for low latency
- 📊 **Confidence scoring** — every prediction includes a transparent confidence value
- 🎨 **Clean UI** — minimal static HTML frontend, no build step required
- 🔍 **Health endpoint** — `/health` for deployment monitoring
- ☁️ **Cloud-ready** — deployed on Render with models hosted on Hugging Face Hub

---

## 🧠 Models

| Language | Model | Hugging Face |
|----------|-------|--------------|
| 🇬🇧 English | DistilBERT (fine-tuned) | [`raed-7/fake-review-distilbert-en`](https://huggingface.co/raed-7/fake-review-distilbert-en) |
| 🇸🇦 Arabic | CamelBERT-Mix (fine-tuned) | [`raed-7/fake-review-camelbert-ar`](https://huggingface.co/raed-7/fake-review-camelbert-ar) |

Models are loaded remotely at application startup and cached for reuse.

---

## 📁 Project Structure

```
App-Fake-Review-Detection/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── predictor.py         # Inference logic & language routing
│   ├── schemas.py           # Pydantic request/response models
│   └── static/
│       └── index.html       # Frontend UI
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment config
├── runtime.txt               # Python version pin
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Raed-7/App-Fake-Review-Detection.git
cd App-Fake-Review-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the server

```bash
uvicorn app.main:app --reload
```

### 4. Open the app

Navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 🔌 API

### `POST /predict`

Submit a review and receive a classification.

**Request body:**

```json
{
  "text": "Best product ever!!! Amazing quality, fast shipping, 5 stars!!!"
}
```

**Response:**

```json
{
  "label": "Fake",
  "confidence": 99.9,
  "language": "English",
  "model": "DistilBERT",
  "processing_time_ms": 1247
}
```

### `GET /health`

Returns model-load status for deployment health checks.

```json
{
  "status": "ok",
  "models_loaded": true
}
```

---

## 🌐 Live Demo

Try the deployed application here:

### 👉 **[https://fake-review-detector-eajs.onrender.com/](https://fake-review-detector-eajs.onrender.com/)**


---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Uvicorn |
| ML / NLP | Transformers, PyTorch, scikit-learn |
| Frontend | Static HTML / CSS / JS |
| Deployment | Render |
| Model Hosting | Hugging Face Hub |

---

## 📚 Research Context

This application is the deployment artefact of a final-year BSc dissertation at the **University of Leeds**:

> *Detecting Fake E-Commerce Reviews in English and Arabic Using Classical and Transformer Models* (2025/26)

The accompanying research compares classical TF-IDF baselines (Logistic Regression, Linear SVM, Random Forest) against fine-tuned transformer models (BERT, DistilBERT, CamelBERT, AraBERT) on bilingual fake review detection.

**Key results:**
- DistilBERT (English): macro-F1 = **0.9813**
- CamelBERT (Arabic): macro-F1 = **0.8582**

---

## 👤 Author

**Raed Alshammari**
BSc Computer Science with Artificial Intelligence
University of Leeds, 2025/26

---

<div align="center">

⭐ **If you found this useful, please consider starring the repository!**

</div>

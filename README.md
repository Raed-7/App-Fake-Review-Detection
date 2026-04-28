# 

# Bilingual Fake Review Detector (English + Arabic)

A FastAPI web application that classifies product reviews as **Fake** or **Real** for both **English** and **Arabic**.

It automatically detects the input language using a Unicode-script heuristic and routes the text to the correct model.

## Features

- English + Arabic support (automatic routing)
- Real/Fake label + confidence score
- FastAPI backend + simple static HTML frontend
- Models loaded once at startup for low-latency inference
- Health endpoint for deployment checks

## Project structure

app/
- main.py
- predictor.py
- schemas.py
- static/

- models/
- english/
- arabic/
- requirements.txt

## Run locally

Install dependencies:
```bash
pip install -r requirements.txt
```
- Start the server from the repository root:
```bash
uicorn app.main:app --reload 
```
Open: http://127.0.0.1:8000 

API:
POST /predict  
Request body:
```bash
{"text": "your review text here"}
```
Response includes:
- predicted label 
- confidence score 
- detected language 
- model name 
- processing time (ms)
GET /health  
- Returns model-load status.
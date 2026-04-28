# ── Adversarial Claim Scoring Engine — Dockerfile ──────────────────────────
# Multi-stage build: lean production image for Cloud Run

FROM python:3.11-slim AS base

# System dependencies (for pdfplumber, chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependency install ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY backend/ ./backend/
COPY app.py .

# ChromaDB persistence directory
RUN mkdir -p /app/chroma_db

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONPATH=/app/backend

# ── Entrypoint ────────────────────────────────────────────────────────────────
EXPOSE 8080
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

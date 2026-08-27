# Build from the project root: docker build -f RENDER/Dockerfile .
FROM python:3.12-slim

WORKDIR /app

# libgomp1 is required by LightGBM at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer cached unless requirements change)
COPY RENDER/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY pipeline/ ./pipeline/

# Copy serving layer (app.py, schemas.py, index.html)
COPY serving/ ./serving/

# Override predictor.py with the HuggingFace-aware version
COPY RENDER/predictor_hf.py ./serving/predictor.py

WORKDIR /app/serving

EXPOSE 8000

# Render sets $PORT automatically; fall back to 8000 for local testing
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

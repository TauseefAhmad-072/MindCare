FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API files
COPY api/ ./api/
COPY models/ ./models/

WORKDIR /app/api

ENV MODELS_DIR=/app/models
ENV PORT=5000

EXPOSE 5000

CMD ["python", "app.py"]

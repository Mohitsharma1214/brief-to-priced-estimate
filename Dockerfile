# Multi-stage production Dockerfile for Brief to Priced Estimate service
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for OpenCV headless execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application source code and data
COPY app/ ./app/
COPY inventory/ ./inventory/
COPY evaluation/ ./evaluation/
COPY docs/ ./docs/
COPY EVALUATION.md README.md ./

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INVENTORY_PATH="/app/inventory/products.json"

EXPOSE 8000

# Healthcheck for container orchestration (Docker Compose / ECS / Kubernetes)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

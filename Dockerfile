# ── Stage 1: Build ────────────────────────────────────────────
# Install dependencies in a separate stage to keep final image small
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
# Lean final image — only what's needed to run
FROM python:3.11-slim AS runtime

WORKDIR /app

# Security: run as non-root user
RUN useradd -m -u 1001 raguser

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed && \
    chown -R raguser:raguser /app

USER raguser

# Tell Python not to buffer output (so logs appear immediately)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.11-slim

# -------------------------------
# 1. System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    fluidsynth \
    fluid-soundfont-gm \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 2. Pip hardening
# -------------------------------
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=100

# -------------------------------
# 3. Create non-root user
# -------------------------------
RUN useradd -m appuser

# -------------------------------
# 4. Working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 5. Install dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --retries 10 -r requirements.txt

# -------------------------------
# 6. Copy application code
# -------------------------------
COPY app/ app/
COPY config.json /app/data/config.json

# -------------------------------
# 7. Fix ownership (CRITICAL)
# -------------------------------
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app

# -------------------------------
# 8. Drop root privileges
# -------------------------------
USER appuser

# -------------------------------
# 9. Expose API port
# -------------------------------
EXPOSE 8000

# -------------------------------
# 10. Start FastAPI
# -------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

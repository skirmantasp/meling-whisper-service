# ---------------------------------------------------------------------------
# Stage 1 — convert NB-Whisper (HuggingFace transformers) → CTranslate2 format
# This stage requires torch + transformers but they are NOT copied to stage 2,
# keeping the runtime image lean.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS converter

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    ctranslate2

ARG WHISPER_MODEL=NbAiLab/nb-whisper-medium
RUN ct2-transformers-converter \
    --model "${WHISPER_MODEL}" \
    --output_dir /models/nb-whisper-medium-ct2 \
    --quantization int8 \
    --force

# ---------------------------------------------------------------------------
# Stage 2 — lean runtime (no torch, no transformers)
# ---------------------------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

# faster-whisper uses CTranslate2 for inference — no torch required at runtime.
RUN pip install --no-cache-dir faster-whisper flask

# Copy the pre-converted model from the builder stage.
COPY --from=converter /models /models

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .

RUN chmod +x /app/start.sh
RUN mkdir -p /tmp/whisper-uploads /tmp/whisper-output

ENV PORT=3000
ENV WHISPER_MODEL=NbAiLab/nb-whisper-medium
ENV WHISPER_MODEL_PATH=/models/nb-whisper-medium-ct2

EXPOSE 3000
CMD ["/app/start.sh"]

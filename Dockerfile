FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU build of PyTorch keeps the image small; transformers drives NB-Whisper.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers accelerate soundfile librosa

# Pre-download the NB-Whisper Norwegian model so the first request is fast.
ARG WHISPER_MODEL=NbAiLab/nb-whisper-medium
RUN python3 -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='${WHISPER_MODEL}')"

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN mkdir -p /tmp/whisper-uploads /tmp/whisper-output

ENV PORT=3000
ENV WHISPER_MODEL=NbAiLab/nb-whisper-medium

EXPOSE 3000
CMD ["node", "server.js"]

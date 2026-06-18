FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install openai-whisper

ARG WHISPER_MODEL=medium
RUN python3 -c "import whisper; whisper.load_model('${WHISPER_MODEL}')"

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN mkdir -p /tmp/whisper-uploads /tmp/whisper-output

ENV PORT=3000
ENV WHISPER_MODEL=medium

EXPOSE 3000
CMD ["node", "server.js"]

# meling-whisper-service

GDPR-compliant audio transcription service for **Sjødin Meling** law firm, built on
the open-source [OpenAI Whisper](https://github.com/openai/whisper) model (not the
hosted API). The service exposes a small Express API for transcribing audio recordings
and searching within the resulting transcripts.

## GDPR compliance

All processing happens on EU infrastructure — **Railway EU-West (Amsterdam)**. Audio is
transcribed locally by the open-source Whisper model, so **no audio or transcript data
ever leaves the EU** and nothing is sent to a third-party API. Temporary upload and
output files are deleted **immediately after each request is processed**.

## Supported audio formats

`mp3`, `wav`, `mp4`, `m4a`, `ogg`, `webm`, `flac` — maximum file size **500 MB**.

## Model options

The model is selected via the `WHISPER_MODEL` environment variable (default `medium`).

| Model    | Parameters | Relative speed | Accuracy        | Notes                              |
| -------- | ---------- | -------------- | --------------- | ---------------------------------- |
| `tiny`   | 39 M       | ~32x           | Lowest          | Fastest, drafts only               |
| `base`   | 74 M       | ~16x           | Low             | Quick previews                     |
| `small`  | 244 M      | ~6x            | Moderate        | Good speed/accuracy balance        |
| `medium` | 769 M      | ~2x            | High            | **Default** — recommended          |
| `large`  | 1550 M     | 1x             | Highest         | Best accuracy, slowest / most RAM  |

Default transcription language is **Norwegian** (`no`). It can be overridden per request
by sending a `language` field in the `/transcribe` form data.

## Environment variables

| Variable        | Default  | Description                          |
| --------------- | -------- | ------------------------------------ |
| `PORT`          | `3000`   | HTTP port the server listens on      |
| `WHISPER_MODEL` | `medium` | Whisper model to load                |

## API

### `GET /health`

Returns service status and metadata.

**Response**

```json
{
  "status": "ok",
  "service": "meling-whisper-service",
  "model": "medium",
  "default_language": "no",
  "gdpr_note": "All audio is processed in-memory on EU infrastructure ...",
  "timestamp": "2026-06-18T10:00:00.000Z"
}
```

### `POST /transcribe`

Accepts a multipart form upload and returns a structured transcription.

**Request** — `multipart/form-data`

| Field      | Type   | Required | Description                              |
| ---------- | ------ | -------- | ---------------------------------------- |
| `audio`    | file   | yes      | The audio file to transcribe             |
| `language` | string | no       | ISO language code (defaults to `no`)     |

```bash
curl -X POST http://localhost:3000/transcribe \
  -F "audio=@interview.mp3" \
  -F "language=no"
```

**Response**

```json
{
  "text": "Full transcription text ...",
  "language": "no",
  "duration_seconds": 184.5,
  "segments": [
    {
      "id": 0,
      "start": "00:00:00",
      "end": "00:00:05",
      "start_seconds": 0.0,
      "end_seconds": 5.2,
      "text": "God morgen, takk for at du kom."
    }
  ],
  "metadata": {
    "original_filename": "interview.mp3",
    "model_used": "medium",
    "processed_at": "2026-06-18T10:00:00.000Z",
    "gdpr_note": "All audio is processed in-memory on EU infrastructure ..."
  }
}
```

### `POST /search`

Searches within a previously returned transcript and returns matching segments with
surrounding context.

**Request** — `application/json`

| Field        | Type   | Required | Description                                    |
| ------------ | ------ | -------- | ---------------------------------------------- |
| `transcript` | object | yes      | A full transcription result from `/transcribe` |
| `query`      | string | yes      | The text to search for (case-insensitive)      |

```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": { "segments": [ ... ] },
    "query": "kontrakt"
  }'
```

**Response**

```json
{
  "query": "kontrakt",
  "match_count": 1,
  "matches": [
    {
      "id": 12,
      "start": "00:01:30",
      "end": "00:01:35",
      "start_seconds": 90.0,
      "end_seconds": 95.0,
      "text": "Vi diskuterte kontrakten i detalj.",
      "context": {
        "before": "Møtet startet presis.",
        "after": "Begge parter var enige."
      }
    }
  ]
}
```

## Running locally

Requires Python 3 with `openai-whisper` installed and `ffmpeg` available on `PATH`.

```bash
npm install
npm start
```

## Deployment

Deployed via Docker to Railway EU-West (Amsterdam). See `Dockerfile` and `railway.toml`.
The Whisper model is pre-downloaded at image build time so the first request is fast.

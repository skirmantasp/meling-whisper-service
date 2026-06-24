# meling-whisper-service

GDPR-compliant audio transcription service for **Sjødin Meling** law firm, built on
the open-source [NB-Whisper](https://huggingface.co/NbAiLab/nb-whisper-medium) model
(not a hosted API). NB-Whisper is trained on ~66,000 hours of Norwegian audio and
reaches a word error rate of ~2.3% on Norwegian, versus ~14.6% for the generic Whisper
`medium` model. The service exposes a small Express API for transcribing audio
recordings and searching within the resulting transcripts.

## GDPR compliance

All processing happens on EU infrastructure — **Railway EU-West (Amsterdam)**. Audio is
transcribed locally by the open-source NB-Whisper model, so **no audio or transcript
data ever leaves the EU** and nothing is sent to a third-party transcription API.
Temporary upload and output files are deleted **immediately after each job is
processed**. (Optional completion emails are sent via Resend — see below.)

## Supported audio formats

`mp3`, `wav`, `mp4`, `m4a`, `ogg`, `webm`, `flac` — maximum file size **500 MB**.

## Model options

The model is selected via the `WHISPER_MODEL` environment variable (default
`NbAiLab/nb-whisper-medium`). Any NB-Whisper HuggingFace checkpoint can be used:

| Model                       | Parameters | Relative speed | Accuracy   | Notes                            |
| --------------------------- | ---------- | -------------- | ---------- | -------------------------------- |
| `NbAiLab/nb-whisper-tiny`   | 39 M       | Fastest        | Lower      | Quick drafts                     |
| `NbAiLab/nb-whisper-base`   | 74 M       | Fast           | Moderate   | Previews                         |
| `NbAiLab/nb-whisper-small`  | 244 M      | Medium         | High       | Good speed/accuracy balance      |
| `NbAiLab/nb-whisper-medium` | 769 M      | Slower         | Very high  | **Default** — recommended        |
| `NbAiLab/nb-whisper-large`  | 1550 M     | Slowest        | Highest    | Best accuracy, most RAM          |

Default transcription language is **Norwegian** (`no`). It can be overridden per request
by sending a `language` field in the `/transcribe` form data.

## Environment variables

| Variable            | Default                     | Description                                            |
| ------------------- | --------------------------- | ------------------------------------------------------ |
| `PORT`              | `3000`                      | HTTP port the server listens on                        |
| `WHISPER_MODEL`     | `NbAiLab/nb-whisper-medium` | NB-Whisper HuggingFace checkpoint to load              |
| `RESEND_API_KEY`    | _(unset)_                   | Resend API key — enables completion emails when set    |
| `RESEND_FROM_EMAIL` | _(unset)_                   | Verified sender address for Resend notifications       |

If `RESEND_API_KEY` / `RESEND_FROM_EMAIL` are not set, email notifications are silently
skipped and transcription works as normal.

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

Accepts a multipart form upload, starts a background transcription **job**, and returns
the job ID immediately. Poll `GET /status/:job_id` for the result. If an `email` is
provided, a notification is also sent when the job finishes.

**Request** — `multipart/form-data`

| Field      | Type   | Required | Description                                              |
| ---------- | ------ | -------- | ------------------------------------------------------- |
| `audio`    | file   | yes      | The audio file to transcribe                            |
| `language` | string | no       | ISO language code (defaults to `no`)                    |
| `email`    | string | no       | If set, a completion/failure email is sent via Resend   |

```bash
curl -X POST http://localhost:3000/transcribe \
  -F "audio=@interview.mp3" \
  -F "language=no" \
  -F "email=advokat@example.no"
```

**Response**

```json
{ "job_id": "f1e2d3c4-...", "status": "processing" }
```

On success, the requester receives an email with subject **"Transkripsjon ferdig ✓"**
containing the job ID, processing time, and a 300-character transcript preview. On
failure, the subject is **"Transkripsjon feilet"** with the job ID and error message.

### `GET /status/:job_id`

Returns the current state of a transcription job. Jobs expire one hour after creation.

**Response — still processing**

```json
{ "status": "processing" }
```

**Response — done**

```json
{
  "status": "done",
  "result": {
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
      "model_used": "NbAiLab/nb-whisper-medium",
      "processed_at": "2026-06-18T10:00:00.000Z",
      "gdpr_note": "All audio is processed in-memory on EU infrastructure ..."
    }
  }
}
```

**Response — error**

```json
{ "status": "error", "error": "Transcription failed: ..." }
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

Requires Python 3 with `transformers` and `torch` installed, plus `ffmpeg` available on
`PATH`. Transcription runs through `transcribe.py`, which loads the NB-Whisper model via
the transformers pipeline.

```bash
pip install transformers torch
npm install
npm start
```

To enable completion emails locally, set `RESEND_API_KEY` and `RESEND_FROM_EMAIL` (e.g.
in a `.env` file).

## Deployment

Deployed via Docker to Railway EU-West (Amsterdam). See `Dockerfile` and `railway.toml`.
The Whisper model is pre-downloaded at image build time so the first request is fast.

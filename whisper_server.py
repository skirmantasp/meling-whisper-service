#!/usr/bin/env python3
"""Persistent HTTP server that serves faster-whisper transcriptions.

Loads the CTranslate2-format NB-Whisper model ONCE at startup and keeps it
in memory. Each POST /transcribe request takes an audio file path and returns
a JSON transcript. A threading.Lock ensures only one inference runs at a time,
which is required because WhisperModel is not thread-safe for concurrent calls.

Listens on 127.0.0.1:8765 (localhost only — never exposed externally).
"""

import json
import os
import threading

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

app = Flask(__name__)
_model_lock = threading.Lock()

MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", "/models/nb-whisper-large-v3-ct2")

print(f"[whisper-server] Loading model from {MODEL_PATH} ...", flush=True)
model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
print("[whisper-server] Model ready.", flush=True)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json(force=True)
    input_path = data["input"]
    language = data.get("language", "no")
    # Optional comma-separated names/places/terms supplied by the caller. Passed
    # to faster-whisper as initial_prompt so the model is primed to recognize
    # proper nouns (e.g. "Observatøren") rather than hallucinating similar-
    # sounding phrases.
    context = (data.get("context") or "").strip()

    transcribe_kwargs = {
        "language": language,
        "beam_size": 5,
        "vad_filter": True,
    }
    if context:
        transcribe_kwargs["initial_prompt"] = context

    with _model_lock:
        segments_iter, info = model.transcribe(input_path, **transcribe_kwargs)

        segments = []
        text_parts = []
        for i, seg in enumerate(segments_iter):
            text = seg.text.strip()
            segments.append({
                "id": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
                # Per-segment confidence signals used by the UI to flag uncertain
                # transcription (low avg log-prob or likely no speech).
                "avg_logprob": float(seg.avg_logprob),
                "no_speech_prob": float(seg.no_speech_prob),
            })
            if text:
                text_parts.append(text)

    return jsonify({
        "text": " ".join(text_parts),
        "language": language,
        "segments": segments,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, threaded=True)

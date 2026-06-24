#!/usr/bin/env python3
"""Transcribe an audio file with NB-Whisper via the HuggingFace transformers pipeline.

NbAiLab/nb-whisper-* models are trained on ~66,000 hours of Norwegian audio and
reach far lower WER on Norwegian than the generic Whisper checkpoints. They are
transformers checkpoints (not the openai-whisper CLI format), so inference runs
through the `automatic-speech-recognition` pipeline here.

The output JSON is written in the exact shape the Node service expects:

    {
      "text": "<full transcript>",
      "language": "<language code>",
      "segments": [
        {"id": 0, "start": 0.0, "end": 5.2, "text": "..."},
        ...
      ]
    }
"""

import argparse
import json
import sys

import torch
from transformers import pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="NB-Whisper transcription helper")
    parser.add_argument("--input", required=True, help="Path to the input audio file")
    parser.add_argument("--model", required=True, help="HuggingFace model id, e.g. NbAiLab/nb-whisper-medium")
    parser.add_argument("--language", default="no", help="Transcription language code (default: no)")
    parser.add_argument("--output", required=True, help="Path to write the result JSON")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    torch_dtype = torch.float16 if use_cuda else torch.float32

    asr = pipeline(
        task="automatic-speech-recognition",
        model=args.model,
        device=device,
        torch_dtype=torch_dtype,
        chunk_length_s=28,
        return_timestamps=True,
    )

    result = asr(
        args.input,
        return_timestamps=True,
        generate_kwargs={"task": "transcribe", "language": args.language},
    )

    segments = []
    for index, chunk in enumerate(result.get("chunks", [])):
        timestamp = chunk.get("timestamp") or (None, None)
        start, end = timestamp[0], timestamp[1]
        start_seconds = float(start) if start is not None else 0.0
        # The final chunk can have a null end timestamp; fall back to its start.
        end_seconds = float(end) if end is not None else start_seconds
        segments.append(
            {
                "id": index,
                "start": start_seconds,
                "end": end_seconds,
                "text": (chunk.get("text") or "").strip(),
            }
        )

    output = {
        "text": (result.get("text") or "").strip(),
        "language": args.language,
        "segments": segments,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

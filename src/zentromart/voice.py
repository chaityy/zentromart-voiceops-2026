from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def voice_turn(audio_path: Path, agent: Any, output_dir: Path) -> dict[str, Any]:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient

    stt = GnaniSTTClient()
    tts = GnaniTTSClient()
    transcript = stt.transcribe(str(audio_path), language_code="en-IN")["transcript"]
    if not transcript.strip():
        reply = "Sorry, I didn't catch that. Could you say it again?"
        result = type("EmptyResult", (), {"text": reply, "sources": [], "tool_calls": []})()
    else:
        result = agent.respond(transcript)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    audio_out = output_dir / f"{stem}_reply.wav"
    audio_bytes = tts.synthesize(result.text, voice="Kaveri", model="timbre-v2.5", language="en-IN")
    audio_out.write_bytes(audio_bytes)
    record = {"input_file": audio_path.name, "transcript": transcript, "reply": result.text,
              "retrieved_sections": result.sources, "tool_calls": result.tool_calls,
              "reply_audio": audio_out.name}
    (output_dir / f"{stem}.txt").write_text(f"Transcript: {transcript}\n\nReply: {result.text}\n\nSources: {', '.join(result.sources)}\n", encoding="utf-8")
    (output_dir / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def run_batch(agent: Any, audio_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    files = sorted(p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO)
    if not files:
        raise ValueError(f"No supported audio files found in {audio_dir}")
    results = [voice_turn(path, agent, output_dir) for path in files]
    (output_dir / "batch_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results

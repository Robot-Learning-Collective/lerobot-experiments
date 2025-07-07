#!/usr/bin/env python3
"""
Hold SPACE → record via device 4 at 48 kHz → send WAV to OpenAI → print transcript.
Release SPACE to stop recording; loop forever.
"""

from __future__ import annotations
import pathlib, queue, threading, time, json, sys

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from pynput import keyboard
from openai import OpenAI, OpenAIError

# ── AUDIO / ASR SETTINGS ───────────────────────────────────────────────
INPUT_ID   = 4                # proven-good microphone device
FS         = 48_000           # device’s native rate
MODEL_NAME = "gpt-4o-transcribe"
LANG_HINT  = "en"
TEMPERATURE = 0.0
# ───────────────────────────────────────────────────────────────────────

sd.default.device = (INPUT_ID, None)

# shared state between listener and worker threads
record_q:   queue.Queue[np.ndarray] = queue.Queue()
is_recording = threading.Event()
stream: sd.InputStream | None = None

# ── THREAD: start stream and push frames into Queue ────────────────────
def recorder_thread():
    global stream
    def callback(indata, _frames, _time, status):
        if status:
            print("⚠️", status, file=sys.stderr)
        if is_recording.is_set():
            record_q.put(indata.copy())
    stream = sd.InputStream(
        samplerate=FS, channels=1, dtype="float32",
        callback=callback, blocksize=0)
    stream.start()
    print("🎙️  Recording… (release SPACE to stop)")

# ── THREAD: stop, assemble WAV, transcribe ─────────────────────────────
def transcriber_thread():
    global stream
    if stream is None:
        return
    stream.stop(); stream.close()

    # Gather frames
    frames = []
    while not record_q.empty():
        frames.append(record_q.get())
    if not frames:
        print("(no audio captured)")
        return
    audio_f32 = np.concatenate(frames).squeeze()
    peak = float(np.abs(audio_f32).max())
    print(f"• Peak {peak:.4f}")

    # Convert to int16 & save
    int16 = np.int16(np.clip(audio_f32, -1, 1) * 32767)
    wav_path = pathlib.Path("/tmp/voice_send.wav")
    write(wav_path, FS, int16)

    # Transcribe
    client = OpenAI()
    try:
        with wav_path.open("rb") as f:
            resp = client.audio.transcriptions.create(
                model=MODEL_NAME,
                file=f,
                language=LANG_HINT,
                temperature=TEMPERATURE,
                response_format="json",
            )
        print("📝 Transcript:", resp.text)
    except OpenAIError as e:
        print("⚠️  OpenAI error:", e, file=sys.stderr)
    finally:
        wav_path.unlink(missing_ok=True)

# ── KEYBOARD HOOKS ─────────────────────────────────────────────────────
def on_press(key):
    if key == keyboard.Key.space and not is_recording.is_set():
        is_recording.set()
        threading.Thread(target=recorder_thread, daemon=True).start()

def on_release(key):
    if key == keyboard.Key.space and is_recording.is_set():
        is_recording.clear()
        threading.Thread(target=transcriber_thread, daemon=True).start()

# ── MAIN LOOP ──────────────────────────────────────────────────────────
def main():
    print("Hold SPACE to speak; release to transcribe.  Ctrl-C to quit.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
        l.join()

if __name__ == "__main__":
    main()

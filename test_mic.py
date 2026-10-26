"""Temporary diagnostic: see what Vosk hears from your microphone, live.

    python test_mic.py                 # list devices, then listen on the default mic
    python test_mic.py --device 1      # listen on a specific device ID
    python test_mic.py --list          # just list devices
    python test_mic.py --seconds 10    # stop by itself after 10 seconds
"""
import argparse
import json
import queue
import sys
import time

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model

from ears import SAMPLE_RATE, Ears, ensure_model

BLOCK_FRAMES = 2000  # 125 ms


def list_input_devices():
    default_in = sd.default.device[0]
    hostapis = sd.query_hostapis()
    print("Input devices (ID | name | host API | channels | default rate):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "  <- default" if i == default_in else ""
            print(
                f"  [{i:2d}] {d['name']} | {hostapis[d['hostapi']]['name']} | "
                f"{d['max_input_channels']} ch | {d['default_samplerate']:.0f} Hz{mark}"
            )
    print()


def level_bar(block, width=20):
    """Tiny volume meter so you can tell the mic is picking up sound at all."""
    rms = np.sqrt(np.mean(np.frombuffer(block, dtype=np.int16).astype(np.float32) ** 2))
    filled = min(width, int(rms / 3000 * width))
    return "#" * filled + "." * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="Live Vosk transcription from the microphone")
    parser.add_argument("--device", type=int, help="input device ID (see the list)")
    parser.add_argument("--list", action="store_true", help="only list input devices")
    parser.add_argument("--seconds", type=float, help="stop automatically after N seconds")
    args = parser.parse_args()

    list_input_devices()
    if args.list:
        return

    device = args.device if args.device is not None else sd.default.device[0]
    print(f"Using device [{device}] {sd.query_devices(device)['name']}")
    print("Loading speech model...")
    rec = KaldiRecognizer(Model(str(ensure_model())), SAMPLE_RATE)
    print("Listening. Speak now (try: 'max', 'stop', or any sentence). Ctrl+C to quit.\n")

    audio = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        if status:
            print(f"\n[audio status: {status}]", file=sys.stderr)
        audio.put(bytes(indata))

    started = time.monotonic()
    last_line_len = 0
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_FRAMES, device=device,
            channels=1, dtype="int16", callback=on_audio,
        ):
            while args.seconds is None or time.monotonic() - started < args.seconds:
                try:
                    block = audio.get(timeout=0.5)
                except queue.Empty:
                    print("\n[no audio arriving from this device]")
                    continue

                if rec.AcceptWaveform(block):  # end of a sentence
                    text = json.loads(rec.Result()).get("text", "")
                    print("\r" + " " * last_line_len + "\r", end="")
                    if text:
                        trigger = Ears._trigger_kind(text)
                        note = f"   <-- Bart would react: {trigger!r}" if trigger else ""
                        print(f"FINAL:   {text}{note}")
                    last_line_len = 0
                else:
                    partial = json.loads(rec.PartialResult()).get("partial", "")
                    line = f"[{level_bar(block)}] {partial}"
                    print("\r" + line.ljust(last_line_len), end="", flush=True)
                    last_line_len = len(line)
    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as e:
        print(f"\nCould not open the microphone: {e}")
        print("Try another device ID from the list above, e.g. --device 1")
        return
    print("\nStopped.")


if __name__ == "__main__":
    main()

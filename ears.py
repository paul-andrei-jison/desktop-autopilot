import json
import queue
import re
import threading
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

from audio_config import INPUT  # also sets sounddevice's default devices on import

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_DIR = Path(__file__).parent / "models" / "vosk"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"

SAMPLE_RATE = 16000
BLOCK_FRAMES = 2000  # 125 ms of audio per read: responsive wake/stop detection

TRIGGER_COOLDOWN_S = 1.0  # ignore repeat triggers right after one fires
WAKE_WORD = "max"
# Words the small model sometimes outputs for "max". Several are ordinary words
# ("marks"), so they only count at the START of an utterance ("Max, ..." / "Hey max, ...").
# The exact word "max" counts anywhere.
MAX_VARIANTS = {"macs", "mack", "macks", "marks", "mac", "lax"}
WAKE_WORDS = {WAKE_WORD} | MAX_VARIANTS
GREETINGS = {"hey", "hay", "hi", "okay", "ok"}

SetLogLevel(-1)  # silence Vosk's console logging


def ensure_model(name=MODEL_NAME, model_dir=MODEL_DIR):
    """Download and unpack the Vosk model if it isn't there; return its folder."""
    target = model_dir / name
    if target.is_dir():
        return target
    model_dir.mkdir(parents=True, exist_ok=True)
    archive = model_dir / f"{name}.zip.part"
    print(f"[downloading speech model {name} (~40 MB) ...]")
    with urlopen(MODEL_URL) as resp, open(archive, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    with zipfile.ZipFile(archive) as z:
        z.extractall(model_dir)
    archive.unlink()
    return target


class _Capture:
    """Records one spoken command using the full-vocabulary recognizer."""

    def __init__(self, model, first_word_timeout, max_seconds):
        self.rec = KaldiRecognizer(model, SAMPLE_RATE)
        self.first_word_timeout = first_word_timeout
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.heard_speech = False
        self.finished = False
        self.result = queue.Queue(maxsize=1)

    def _finish(self, text):
        self.finished = True
        self.result.put(text or None)

    def feed(self, data):
        elapsed = time.monotonic() - self.started
        if self.rec.AcceptWaveform(data):  # Vosk detected the end of an utterance
            text = json.loads(self.rec.Result()).get("text", "").strip()
            if text:
                return self._finish(text)
        elif json.loads(self.rec.PartialResult()).get("partial"):
            self.heard_speech = True

        if not self.heard_speech and elapsed > self.first_word_timeout:
            self._finish(None)  # nobody said anything
        elif elapsed > self.max_seconds:
            self._finish(json.loads(self.rec.FinalResult()).get("text", "").strip())


class Ears:
    """Always-on offline microphone listener.

    In the background it listens for the wake word "max" / "stop" and reports them
    through on_trigger(kind). listen_for_command() temporarily switches the same
    microphone stream over to full speech recognition and returns the transcript.

    `source` is for tests: a generator of 16 kHz int16 audio blocks (bytes) used
    in place of the real microphone.
    """

    def __init__(self, on_trigger, source=None):
        self.on_trigger = on_trigger
        self._source = source
        self.model = Model(str(ensure_model()))
        self._wake = KaldiRecognizer(self.model, SAMPLE_RATE)
        self._capture = None
        self._last_trigger = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    # --- background loop -------------------------------------------------

    def _blocks(self):
        if self._source is not None:
            yield from self._source
            return
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_FRAMES, device=INPUT,
            channels=1, dtype="int16",
        ) as stream:
            while True:
                data, _overflowed = stream.read(BLOCK_FRAMES)
                yield bytes(data)

    def _run(self):
        was_capturing = False
        try:
            for data in self._blocks():
                cap = self._capture
                if cap is not None and not cap.finished:
                    cap.feed(data)
                    was_capturing = True
                    continue
                if was_capturing:  # back to wake mode: drop leftover state
                    self._wake.Reset()
                    was_capturing = False
                self._detect_trigger(data)
        except Exception as e:
            print(f"[microphone error: {e}]")

    @staticmethod
    def _trigger_kind(text):
        words = text.lower().split()
        if WAKE_WORD in words:
            return "max"
        first = 1 if words[:1] and words[0] in GREETINGS else 0
        if len(words) > first and words[first] in MAX_VARIANTS:
            return "max"
        if "stop" in words:
            return "stop"
        return None

    def _detect_trigger(self, data):
        # Partial results let us react while the phrase is still being spoken, so
        # "max, what time is it" starts working before the sentence ends.
        if self._wake.AcceptWaveform(data):
            text = json.loads(self._wake.Result()).get("text", "")
        else:
            text = json.loads(self._wake.PartialResult()).get("partial", "")
        kind = self._trigger_kind(text)
        if kind is None:
            return

        now = time.monotonic()
        self._wake.Reset()
        if now - self._last_trigger < TRIGGER_COOLDOWN_S:
            return
        self._last_trigger = now
        self.on_trigger(kind)

    # --- command capture -------------------------------------------------

    def listen_for_command(self, first_word_timeout=5.0, max_seconds=20.0):
        """Block until the user speaks a full command; None if they stay silent."""
        cap = _Capture(self.model, first_word_timeout, max_seconds)
        self._capture = cap
        try:
            return cap.result.get()
        finally:
            self._capture = None


if __name__ == "__main__":
    def show(kind):
        print(f"  -> trigger heard: {kind!r}")

    ears = Ears(show)
    ears.start()
    print("Say 'max' (or 'stop'). Ctrl+C to quit.")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

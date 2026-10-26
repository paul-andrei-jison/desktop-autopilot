import json
import queue
import re
import threading
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from vosk import KaldiRecognizer, Model, SetLogLevel

from audio_config import open_with_retry  # also sets sounddevice's default devices on import

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_DIR = Path(__file__).parent / "models" / "vosk"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"

SAMPLE_RATE = 16000
BLOCK_FRAMES = 2000  # 125 ms of audio per read: responsive wake/stop detection

# Command capture: end of speech = this much trailing silence.
END_SILENCE_S = 1.0
MIN_SPEECH_RMS = 250        # int16 RMS below this is never speech
SPEECH_FLOOR_RATIO = 2.5    # ...and speech must be this many times the noise floor
WHISPER_MODEL = "base.en"

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


def _load_whisper():
    """Load faster-whisper once: on the GPU if CUDA is available, else int8 on the CPU."""
    try:
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda = False
    if cuda:
        try:
            return WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"[whisper: CUDA unavailable ({e}); using CPU]")
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


class _Capture:
    """Records one spoken command until the speaker falls silent.

    Endpointing is energy based: the noise floor is tracked as the quietest block seen,
    and speech is anything clearly above it. Whisper does the actual transcription later.
    """

    def __init__(self, first_word_timeout, max_seconds):
        self.first_word_timeout = first_word_timeout
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.blocks = []
        self.noise_floor = None
        self.heard_speech = False
        self.silent_blocks = 0
        self.finished = False
        self.result = queue.Queue(maxsize=1)  # holds float32 audio, or None

    def _finish(self, audio):
        self.finished = True
        self.result.put(audio)

    def feed(self, data):
        if self.finished:
            return
        samples = np.frombuffer(data, dtype=np.int16)
        level = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        self.noise_floor = level if self.noise_floor is None else min(self.noise_floor, level)
        speaking = level > max(MIN_SPEECH_RMS, self.noise_floor * SPEECH_FLOOR_RATIO)
        elapsed = time.monotonic() - self.started

        if speaking:
            self.heard_speech = True
            self.silent_blocks = 0
        elif self.heard_speech:
            self.silent_blocks += 1
        if self.heard_speech:
            self.blocks.append(samples)

        if not self.heard_speech:
            if elapsed > self.first_word_timeout:
                self._finish(None)  # nobody said anything
        elif (self.silent_blocks * BLOCK_FRAMES / SAMPLE_RATE >= END_SILENCE_S
              or elapsed > self.max_seconds):
            audio = np.concatenate(self.blocks).astype(np.float32) / 32768.0
            self._finish(audio)


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
        self.whisper = _load_whisper()
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
        stream = open_with_retry(
            lambda device: sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=BLOCK_FRAMES, device=device,
                channels=1, dtype="int16",
            ),
            "input",
        )
        with stream:
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
        cap = _Capture(first_word_timeout, max_seconds)
        self._capture = cap
        try:
            audio = cap.result.get()
        finally:
            self._capture = None
        if audio is None:
            return None
        # Transcribed here, on the caller's thread, so the mic/wake loop never stalls.
        segments, _info = self.whisper.transcribe(
            audio, language="en", beam_size=5, vad_filter=True, condition_on_previous_text=False
        )
        return " ".join(seg.text.strip() for seg in segments).strip() or None


if __name__ == "__main__":
    # Live test: say "max", then a command; the Whisper transcript is printed.
    fired = threading.Event()

    def show(kind):
        print(f"  -> trigger heard: {kind!r}")
        if kind == "max":
            fired.set()

    ears = Ears(show)
    ears.start()
    print("Say 'max', then your command. Ctrl+C to quit.")
    try:
        while True:
            if fired.wait(0.2):
                fired.clear()
                print("  listening ...")
                t0 = time.monotonic()
                text = ears.listen_for_command(first_word_timeout=5.0)
                print(f"  heard: {text!r}  ({time.monotonic() - t0:.1f}s incl. transcription)")
    except KeyboardInterrupt:
        pass

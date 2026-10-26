import queue
import re
import threading
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import sounddevice as sd
from piper import PiperVoice

from audio_config import OUTPUT  # also sets sounddevice's default devices on import

VOICE_NAME = "en_US-danny-low"  # <lang>_<REGION>-<voice>-<quality>
MODEL_DIR = Path(__file__).parent / "models" / "tts"
VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PAUSE_SECONDS = 0.15  # silence after each sentence: natural pacing + flushes the audio buffer


def ensure_voice(name=VOICE_NAME, model_dir=MODEL_DIR):
    """Download the Piper voice (.onnx + .onnx.json) if missing; return the model path."""
    model_dir.mkdir(parents=True, exist_ok=True)
    lang_region, voice, quality = name.split("-")
    base = f"{VOICE_BASE_URL}/{lang_region.split('_')[0]}/{lang_region}/{voice}/{quality}"
    for ext in (".onnx", ".onnx.json"):
        target = model_dir / f"{name}{ext}"
        if target.exists():
            continue
        print(f"[downloading voice file {target.name} ...]")
        partial = target.with_name(target.name + ".part")
        with urlopen(f"{base}/{name}{ext}?download=true") as resp, open(partial, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        partial.replace(target)  # only appears under its real name once complete
    return model_dir / f"{name}.onnx"


class Mouth:
    """Offline neural text-to-speech (Piper ONNX) played through sounddevice."""

    def __init__(self):
        self.voice = PiperVoice.load(ensure_voice())
        self.sample_rate = self.voice.config.sample_rate

    def synthesize(self, text):
        """Return the sentence as one int16 audio array (plus trailing pause)."""
        parts = [chunk.audio_int16_array for chunk in self.voice.synthesize(text)]
        parts.append(np.zeros(int(self.sample_rate * PAUSE_SECONDS), dtype=np.int16))
        return np.concatenate(parts)

    def speak(self, text):
        """Blocking one-shot speech."""
        if re.search(r"\w", text):
            sd.play(self.synthesize(text), self.sample_rate, device=OUTPUT)
            sd.wait()


def speak(text):
    Mouth().speak(text)


class SpeechQueue:
    """Speaks queued sentences in order, without blocking the caller.

    Two background threads form a pipeline: one synthesizes sentence N+1 while the
    other plays sentence N, so there is no gap between sentences.
        say() -> text queue -> synth thread -> audio queue -> playback thread
    """

    PLAY_CHUNK_S = 0.05  # audio is written in slices this long so interrupt() cuts in fast

    def __init__(self, on_speech_start=None):
        self.on_speech_start = on_speech_start  # called once when the first sentence of a reply is queued
        self._mouth = Mouth()
        self._text = queue.Queue()
        self._audio = queue.Queue()
        self._generation = 0  # bumped by interrupt(); older queued work is discarded
        self._muted = False
        self._started = False
        self._threads = [
            threading.Thread(target=self._synth_worker, daemon=True),
            threading.Thread(target=self._play_worker, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _synth_worker(self):
        while True:
            item = self._text.get()
            try:
                if item is None:
                    self._audio.put(None)
                    return
                generation, sentence = item
                if generation == self._generation:
                    self._audio.put((generation, self._mouth.synthesize(sentence)))
            except Exception as e:  # keep the pipeline alive on a bad sentence
                print(f"[tts error: {e}]")
            finally:
                self._text.task_done()

    def _play_worker(self):
        # One stream stays open for the whole session: no per-sentence startup delay.
        chunk = int(self._mouth.sample_rate * self.PLAY_CHUNK_S)
        with sd.OutputStream(
            samplerate=self._mouth.sample_rate, device=OUTPUT, channels=1, dtype="int16"
        ) as stream:
            while True:
                item = self._audio.get()
                try:
                    if item is None:
                        return
                    generation, audio = item
                    for i in range(0, len(audio), chunk):
                        if generation != self._generation:  # interrupted mid-sentence
                            stream.abort()  # throw away audio already buffered in the device
                            stream.start()
                            break
                        stream.write(audio[i : i + chunk])
                except Exception as e:
                    print(f"[audio error: {e}]")
                finally:
                    self._audio.task_done()

    def say(self, sentence):
        if self._muted:
            return
        # Strip markdown symbols that would be read aloud ("asterisk asterisk").
        sentence = re.sub(r"[*#`_>]+", "", sentence).strip()
        if re.search(r"\w", sentence):
            self._text.put((self._generation, sentence))
            if not self._started:
                self._started = True
                if self.on_speech_start:
                    self.on_speech_start()

    def wait(self):
        """Block until every queued sentence has been synthesized and played."""
        self._text.join()
        self._audio.join()
        self._started = False

    def interrupt(self):
        """Stop talking right now and ignore further say() calls until resume()."""
        self._muted = True
        self._generation += 1

    def resume(self):
        """Allow speech again (call at the start of each new reply)."""
        self._muted = False
        self._started = False

    def close(self):
        self._text.put(None)
        for t in self._threads:
            t.join()


if __name__ == "__main__":
    speech = SpeechQueue()
    speech.say("Hello Andrei! I am Bart, your offline AI assistant.")
    speech.say("This is my new neural voice, streamed one sentence at a time.")
    speech.wait()
    speech.close()

"""Which microphone and speaker Bart uses (IDs come from `python list_devices.py`)."""
import time

import sounddevice as sd

INPUT_DEVICE = 1    # Headset (Soundcore Life Q35) mic, MME
OUTPUT_DEVICE = 3   # 2 - ATLAS HD 236C (AMD High Definition Audio Device), MME

# Windows renumbers devices when hardware connects or disconnects (e.g. the Bluetooth
# headset). If the configured ID no longer holds the expected device, we look it up by name.
INPUT_NAME_HINT = "Soundcore"
OUTPUT_NAME_HINT = "ATLAS HD 236C"


def _resolve(device_id, kind, name_hint):
    """Return a usable device ID for `kind` ('input'/'output'), or None for the system default."""
    channels = f"max_{kind}_channels"
    devices = sd.query_devices()

    if 0 <= device_id < len(devices):
        d = devices[device_id]
        if d[channels] > 0 and name_hint.lower() in d["name"].lower():
            return device_id

    # The ID moved: find the same hardware by name, preferring the same audio API (MME).
    mme = next((i for i, a in enumerate(sd.query_hostapis()) if a["name"] == "MME"), None)
    matches = [
        i for i, d in enumerate(devices)
        if d[channels] > 0 and name_hint.lower() in d["name"].lower()
    ]
    matches.sort(key=lambda i: devices[i]["hostapi"] != mme)
    if matches:
        print(f"[audio] {kind} device {device_id} changed; using [{matches[0]}] {devices[matches[0]]['name']}")
        return matches[0]

    print(f"[audio] {kind} device matching {name_hint!r} not found; using the system default")
    return None


def configure():
    """Resolve both devices and set them as sounddevice's defaults. Returns (input, output)."""
    inp = _resolve(INPUT_DEVICE, "input", INPUT_NAME_HINT)
    out = _resolve(OUTPUT_DEVICE, "output", OUTPUT_NAME_HINT)
    # Fallback for any code that doesn't pass device= explicitly (None keeps the system default).
    sd.default.device = (inp, out)
    return inp, out


def open_with_retry(make_stream, kind, attempts=5, delay=2.0):
    """Open an audio stream, retrying while Bluetooth/HDMI devices finish connecting after login.

    `make_stream(device)` builds the stream; the device ID is re-resolved on every attempt
    because Windows may renumber devices once the headset appears.
    """
    for attempt in range(1, attempts + 1):
        device = INPUT if kind == "input" else OUTPUT
        try:
            return make_stream(device)
        except Exception as e:
            print(f"[audio] {kind} device {device} not ready (attempt {attempt}/{attempts}): {e}")
            if attempt == attempts:
                raise
            time.sleep(delay)
            globals()["INPUT" if kind == "input" else "OUTPUT"] = _resolve(
                INPUT_DEVICE if kind == "input" else OUTPUT_DEVICE,
                kind,
                INPUT_NAME_HINT if kind == "input" else OUTPUT_NAME_HINT,
            )


# Runs once on import, so every module sees the same devices.
INPUT, OUTPUT = configure()

if __name__ == "__main__":
    print(f"input  -> {INPUT}: {sd.query_devices(INPUT)['name'] if INPUT is not None else 'system default'}")
    print(f"output -> {OUTPUT}: {sd.query_devices(OUTPUT)['name'] if OUTPUT is not None else 'system default'}")

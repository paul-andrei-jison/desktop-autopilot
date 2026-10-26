import sounddevice as sd

default_in, default_out = sd.default.device
hostapis = sd.query_hostapis()

print(f"{'ID':>3}  {'In':>3}  {'Out':>3}  {'Host API':<22} Name")
for i, d in enumerate(sd.query_devices()):
    tags = []
    if i == default_in:
        tags.append("default input")
    if i == default_out:
        tags.append("default output")
    note = f"   <- {', '.join(tags)}" if tags else ""
    print(
        f"{i:>3}  {d['max_input_channels']:>3}  {d['max_output_channels']:>3}  "
        f"{hostapis[d['hostapi']]['name']:<22} {d['name']}{note}"
    )

import json
import os
import re
import shutil
import subprocess
import sys
import winreg
from pathlib import Path

from pycaw.pycaw import AudioUtilities

# Friendly names -> (display name, launch targets tried in order).
# A target is a command on PATH, an .exe name registered in Windows, or a URI scheme.
KNOWN_APPS = {
    "vs code": ("Visual Studio Code", ["code"]),
    "vscode": ("Visual Studio Code", ["code"]),
    "visual studio code": ("Visual Studio Code", ["code"]),
    "spotify": ("Spotify", ["spotify.exe", "spotify:"]),
    "chrome": ("Google Chrome", ["chrome.exe"]),
    "google chrome": ("Google Chrome", ["chrome.exe"]),
    "edge": ("Microsoft Edge", ["msedge.exe"]),
    "calculator": ("Calculator", ["calc.exe"]),
    "calc": ("Calculator", ["calc.exe"]),
    "notepad": ("Notepad", ["notepad.exe"]),
    "file explorer": ("File Explorer", ["explorer.exe"]),
    "explorer": ("File Explorer", ["explorer.exe"]),
    "settings": ("Settings", ["ms-settings:"]),
    "task manager": ("Task Manager", ["taskmgr.exe"]),
    "terminal": ("Windows Terminal", ["wt.exe"]),
    "powershell": ("PowerShell", ["powershell.exe"]),
    "command prompt": ("Command Prompt", ["cmd.exe"]),
    "paint": ("Paint", ["mspaint.exe"]),
}

START_MENU_DIRS = [
    Path(os.environ.get("ProgramData", r"C:\ProgramData")) / r"Microsoft\Windows\Start Menu\Programs",
    Path(os.environ.get("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs",
]
APP_PATHS_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
STARTUP_VALUE_NAME = "Bart"

# Launch detached, with no console window flashing up for .cmd/.bat shims like `code`.
_POPEN_FLAGS = (
    subprocess.CREATE_NEW_PROCESS_GROUP
    | subprocess.DETACHED_PROCESS
    | subprocess.CREATE_NO_WINDOW
)


def _normalize(name):
    name = name.lower().strip()
    name = re.sub(r"\b(the|app|application|program)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _is_uri_scheme(target):
    """True if target looks like 'scheme:' and Windows has that URI handler registered."""
    if not re.fullmatch(r"[a-z][a-z0-9+.-]*:", target):
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, target[:-1]) as key:
            winreg.QueryValueEx(key, "URL Protocol")
            return True
    except OSError:
        return False


def _registered_exe(exe):
    """Look up an .exe in the Windows 'App Paths' registry (how Chrome, Word, etc. register)."""
    if not exe.lower().endswith(".exe"):
        exe += ".exe"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, rf"{APP_PATHS_KEY}\{exe}") as key:
                path = winreg.QueryValue(key, None)
                if path and Path(path.strip('"')).exists():
                    return path.strip('"')
        except OSError:
            continue
    return None


def _resolve_target(target):
    """Turn one target string into something launchable: ('exe'|'uri', value) or None."""
    path = shutil.which(target)  # PATH + App Execution Aliases (WindowsApps folder)
    if path:
        return "exe", path
    path = _registered_exe(target)
    if path:
        return "exe", path
    if _is_uri_scheme(target):
        return "uri", target
    return None


def _find_start_menu_shortcut(name):
    """Find a Start Menu .lnk whose name matches (exact match first, then substring)."""
    partial = None
    for base in START_MENU_DIRS:
        if not base.is_dir():
            continue
        for lnk in base.rglob("*.lnk"):
            stem = lnk.stem.lower()
            if stem == name:
                return str(lnk)
            if name in stem and partial is None:
                partial = str(lnk)
    return partial


def _find_store_app(name):
    """Find an installed Store/UWP app via Get-StartApps; return its AppUserModelID."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-StartApps | ConvertTo-Json"],
            capture_output=True, text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
        apps = json.loads(out) if out.strip() else []
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    if isinstance(apps, dict):
        apps = [apps]
    # Matching happens in Python, so the app name never reaches a shell command.
    for app in apps:
        if app["Name"].lower() == name:
            return app["AppID"]
    for app in apps:
        if name in app["Name"].lower():
            return app["AppID"]
    return None


def _launch(kind, value):
    if kind == "exe":
        subprocess.Popen([value], creationflags=_POPEN_FLAGS, close_fds=True)
    else:  # 'uri', 'lnk' and 'store' go through the Windows shell
        os.startfile(value)


def open_application(app_name: str) -> str:
    """Open a Windows application by name, without blocking. Returns a status string."""
    key = _normalize(app_name)
    if not key:
        return "Could not find an application named '' installed on this machine."

    display, targets = KNOWN_APPS.get(key, (app_name.strip().title(), [key.replace(" ", "")]))
    if key not in KNOWN_APPS:
        targets = [key, key.replace(" ", "")]

    try:
        # 1. Direct command / registered exe / URI scheme.
        for target in targets:
            resolved = _resolve_target(target)
            if resolved:
                _launch(*resolved)
                return f"Successfully opened {display}"

        # 2. Start Menu shortcut (covers most desktop installers).
        lnk = _find_start_menu_shortcut(key)
        if lnk:
            _launch("lnk", lnk)
            return f"Successfully opened {display}"

        # 3. Microsoft Store apps.
        app_id = _find_store_app(key)
        if app_id:
            _launch("store", rf"shell:AppsFolder\{app_id}")
            return f"Successfully opened {display}"
    except OSError as e:
        return f"Found {display} but failed to launch it: {e}"

    return f"Could not find an application named {app_name} installed on this machine."


def _endpoint_volume():
    """The master volume control of the default speakers (via Windows Core Audio)."""
    return AudioUtilities.GetSpeakers().EndpointVolume


def set_volume(level_percent: int) -> str:
    """Set the Windows master volume to 0-100 percent."""
    try:
        level = int(level_percent)
    except (TypeError, ValueError):
        return f"Could not set volume: '{level_percent}' is not a number between 0 and 100."
    level = max(0, min(100, level))
    try:
        _endpoint_volume().SetMasterVolumeLevelScalar(level / 100, None)
    except Exception as e:
        return f"Could not set volume: {e}"
    return f"Volume set to {level} percent"


def get_volume() -> int:
    return round(_endpoint_volume().GetMasterVolumeLevelScalar() * 100)


def toggle_mute(mute: bool) -> str:
    """Mute (True) or unmute (False) the main Windows audio output."""
    if isinstance(mute, str):  # models sometimes send "true"/"false" as text
        mute = mute.strip().lower() in ("true", "yes", "1", "mute", "muted")
    try:
        _endpoint_volume().SetMute(1 if mute else 0, None)
    except Exception as e:
        return f"Could not change mute: {e}"
    return "Audio muted" if mute else "Audio unmuted"


def _write_launcher() -> Path:
    """Write launch_max.vbs in the project root and return its path.

    The Run key starts programs in C:\\Windows\\System32, so the VBScript sets the working
    directory explicitly and runs pythonw.exe with window style 0 (hidden) - no console at all.
    """
    root = Path(__file__).resolve().parent
    venv_python = root / "venv" / "Scripts" / "pythonw.exe"
    if venv_python.exists():
        python = venv_python
    else:  # no venv: use the pythonw beside the interpreter running us (python.exe would show a console)
        exe = Path(sys.executable).resolve()
        python = exe.with_name("pythonw.exe") if exe.with_name("pythonw.exe").exists() else exe
    ollama = shutil.which("ollama") or "ollama"
    script = (
        'Set WshShell = CreateObject("WScript.Shell")\r\n'
        f'WshShell.CurrentDirectory = "{root}"\r\n'
        "\r\n"
        "Function OllamaUp()\r\n"
        "    OllamaUp = False\r\n"
        "    On Error Resume Next\r\n"
        '    Set http = CreateObject("WinHttp.WinHttpRequest.5.1")\r\n'
        '    http.Open "GET", "http://127.0.0.1:11434/", False\r\n'
        "    http.SetTimeouts 1000, 1000, 1000, 1000\r\n"
        "    http.Send\r\n"
        "    If Err.Number = 0 Then OllamaUp = True\r\n"
        "End Function\r\n"
        "\r\n"
        "If Not OllamaUp() Then\r\n"
        f'    WshShell.Run """" & "{ollama}" & """ serve", 0, False\r\n'
        "    For i = 1 To 30\r\n"
        "        WScript.Sleep 1000\r\n"
        "        If OllamaUp() Then Exit For\r\n"
        "    Next\r\n"
        "End If\r\n"
        "\r\n"
        f'WshShell.Run """" & "{python}" & """ brain.py", 0, False\r\n'
    )
    vbs = root / "launch_max.vbs"
    vbs.write_text(script, encoding="mbcs", newline="")
    return vbs


def _startup_command():
    """Command line Windows runs at login: wscript.exe running the launcher."""
    return f'wscript.exe "{_write_launcher()}"'


def enable_startup_launch() -> str:
    """Make Bart (brain.py) start automatically when the user logs in to Windows."""
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.SetValueEx(key, STARTUP_VALUE_NAME, 0, winreg.REG_SZ, _startup_command())
    except OSError as e:
        return f"Could not enable launch at startup: {e}"
    return "Bart will now start automatically when Windows boots"


def disable_startup_launch() -> str:
    """Undo enable_startup_launch()."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, STARTUP_VALUE_NAME)
    except FileNotFoundError:
        return "Bart was not set to start automatically"
    except OSError as e:
        return f"Could not disable launch at startup: {e}"
    return "Bart will no longer start automatically"


def _startup_value():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            return winreg.QueryValueEx(key, STARTUP_VALUE_NAME)[0]
    except FileNotFoundError:
        return None


def _test_apps():
    for name in ("notepad", "vs code", "Spotify", "definitely not a real app"):
        print(f"open_application({name!r}) -> {open_application(name)}")


def _test_volume():
    """Change volume/mute, read it back, then restore what the user had."""
    original_volume = get_volume()
    original_mute = bool(_endpoint_volume().GetMute())
    print(f"current volume: {original_volume}%, muted: {original_mute}")
    try:
        print(set_volume(30), "->", f"reads back {get_volume()}%")
        print(set_volume(150), "->", f"clamped, reads back {get_volume()}%")
        print(set_volume("abc"))
        print(toggle_mute(True), "->", f"muted={bool(_endpoint_volume().GetMute())}")
        print(toggle_mute(False), "->", f"muted={bool(_endpoint_volume().GetMute())}")
    finally:
        set_volume(original_volume)
        toggle_mute(original_mute)
    print(f"restored: {get_volume()}%, muted: {bool(_endpoint_volume().GetMute())}")


def _test_startup():
    """Write the Run entry, read it back, then remove it so the test leaves no trace."""
    was_enabled = _startup_value() is not None
    print(enable_startup_launch())
    print("registry value:", _startup_value())
    if was_enabled:
        print("(entry existed before this test; leaving it enabled)")
    else:
        print(disable_startup_launch())
        print("registry value after removal:", _startup_value())


if __name__ == "__main__":
    # `python hands.py` tests volume + startup; add --apps to also open real apps.
    _test_volume()
    _test_startup()
    if "--apps" in sys.argv:
        _test_apps()

import os
import re
import signal
import sys
import threading
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "bart.log"


def _log_exception(exc_type, exc, tb):
    """Append an unhandled exception to bart.log (absolute path, independent of the cwd)."""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("[unhandled exception]\n")
            f.write("".join(traceback.format_exception(exc_type, exc, tb)))
    except OSError:
        pass


# Installed before the heavy imports so a failed import at startup is logged too.
os.chdir(PROJECT_ROOT)  # relative paths (models, config) must not depend on the launch directory
sys.excepthook = _log_exception
threading.excepthook = lambda a: _log_exception(a.exc_type, a.exc_value, a.exc_traceback)

import ollama
from PySide6.QtCore import QMetaObject, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ears import GREETINGS, WAKE_WORDS, Ears
from hands import enable_startup_launch, open_application, set_volume, toggle_mute
from mouth import SpeechQueue
from widget import IDLE, LISTENING, SPEAKING, THINKING, OrbWidget

# Cross-thread signals between the microphone thread and the assistant thread.
wake_event = threading.Event()    # "max" heard while idle
interrupted = threading.Event()   # "max"/"stop" heard while Bart is busy
turn_active = threading.Event()   # Bart is thinking or speaking

ACTIVE_WINDOW_S = 5.0  # how long Bart keeps listening for a (follow-up) command

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": (
                "Open an application on the user's Windows computer. Only call this "
                "when the user explicitly asks to open, launch or start an app."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the app, e.g. 'notepad', 'vs code', 'chrome', 'spotify'",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": (
                "Set the computer's master volume. Only call this when the user asks "
                "to change the volume to a level."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level_percent": {
                        "type": "integer",
                        "description": "Volume from 0 (silent) to 100 (maximum)",
                    }
                },
                "required": ["level_percent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_mute",
            "description": "Mute or unmute the computer's audio. Only call this when the user asks to mute or unmute.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mute": {
                        "type": "boolean",
                        "description": "true to mute, false to unmute",
                    }
                },
                "required": ["mute"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_startup_launch",
            "description": (
                "Make Bart start automatically when Windows boots. Only call this when "
                "the user asks for Bart to launch at startup."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
TOOL_FUNCTIONS = {
    "open_application": open_application,
    "set_volume": set_volume,
    "toggle_mute": toggle_mute,
    "enable_startup_launch": enable_startup_launch,
}
MAX_TOOL_ROUNDS = 3  # safety cap so a confused model can't loop forever

# llama3.2 (3B) calls tools even for plain questions, so each tool is only offered
# when the user's message sounds like a request for it.
TOOL_INTENTS = {
    "open_application": re.compile(r"\b(open|launch|start|run|fire up|bring up)\b", re.I),
    "set_volume": re.compile(r"\b(volume|louder|quieter|softer|turn (it )?(up|down)|sound)\b", re.I),
    "toggle_mute": re.compile(r"\b(mute|unmute|silence|silent)\b", re.I),
    "enable_startup_launch": re.compile(
        r"\b(start-?up|starts? up|boots?|log ?in|sign ?in|auto-?start|"
        r"(start|launch|run|open)s? (automatically|on its own|by itself))\b",
        re.I,
    ),
}


def select_tools(user_input):
    """Return only the tool schemas whose intent pattern matches, or None."""
    names = {n for n, pattern in TOOL_INTENTS.items() if pattern.search(user_input)}
    if "enable_startup_launch" in names:
        names.discard("open_application")  # "start automatically" isn't "start an app"
    return [t for t in TOOLS if t["function"]["name"] in names] or None

# A sentence ends at . ! ? followed by whitespace (so "3.14" isn't split), or at a newline.
SENTENCE_END = re.compile(r"[.!?]+(?=\s)|\n")

MODEL = "llama3.2"
SYSTEM_PROMPT = (
    "You are Bart, a concise, helpful local AI assistant running directly "
    "on the user's Windows 11 machine. You can open applications, change "
    "the volume, mute audio, and set yourself to launch at startup using your "
    "tools, but only when the user explicitly asks. After using a tool, tell the "
    "user in one short sentence what happened. The user wakes you by saying the "
    "wake word 'Max'; your name is still Bart."
)
# Spoken commands. "bye" ends the conversation (Bart goes back to waiting for his
# wake word); "quit"/"exit" shuts the whole program down.
END_WORDS = {"bye", "goodbye", "good bye", "that's all", "thats all"}
QUIT_WORDS = {"exit", "quit", "shut down"}  # a trailing "max" is ignored, see spoken_command()

SUMMARY_TRIGGER = 10  # messages in recent_history (5 full turns)
KEEP_AFTER_SUMMARY = 2  # most recent messages kept verbatim

SUMMARY_PROMPT = (
    "You maintain the long-term memory of an AI assistant. Combine the existing "
    "summary (if any) and the recent conversation into an updated, highly concise "
    "bulleted summary.\n"
    "Use exactly two sections:\n"
    "User facts:\n"
    "- Personal details the user stated about themselves (name, job, projects, "
    "preferences). Write them as explicit facts, e.g. \"The user's name is X\". "
    "Never drop or alter an existing user fact unless the user corrects it.\n"
    "Topics discussed:\n"
    "- One short bullet per topic, without the assistant's explanations.\n"
    "Output only these two sections."
)


def summarize(running_summary, recent_history):
    """Fold recent_history into running_summary and return the new summary."""
    transcript = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in recent_history
    )
    content = (
        f"Existing summary:\n{running_summary or '(none)'}\n\n"
        f"Recent conversation:\n{transcript}"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return response["message"]["content"].strip()


def build_payload(running_summary, recent_history):
    """Assemble system prompt + memory summary + active turns for this call."""
    system = SYSTEM_PROMPT
    if running_summary:
        # Single system message: small models weigh it more than a second one.
        system += (
            "\n\nWhat you remember from earlier in this conversation (treat as "
            "true and use it when the user asks about themselves or past "
            f"topics):\n{running_summary}"
        )
    return [{"role": "system", "content": system}] + recent_history


def stream_reply(payload, speech, tools=None):
    """Stream one model response: print tokens live, queue each full sentence for TTS.

    Returns (text, tool_calls). Tool calls arrive as stream chunks alongside (or
    instead of) text, so they are collected here and executed by the caller.
    """
    buffer = ""
    full_reply = ""
    tool_calls = []
    for chunk in ollama.chat(model=MODEL, messages=payload, tools=tools, stream=True):
        if interrupted.is_set():  # user said "max"/"stop": abandon this reply
            return full_reply.strip(), []
        message = chunk["message"]
        token = message["content"] or ""
        print(token, end="", flush=True)
        full_reply += token
        buffer += token
        tool_calls.extend(message.get("tool_calls") or [])

        while True:
            match = SENTENCE_END.search(buffer)
            if not match:
                break
            speech.say(buffer[: match.end()])
            buffer = buffer[match.end():]

    if not interrupted.is_set():
        speech.say(buffer)  # trailing text with no closing punctuation
    return full_reply.strip(), tool_calls


def run_tool(call):
    """Execute one tool call from the model and return its result string."""
    name = call.function.name
    args = call.function.arguments or {}
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        result = func(**args)
    except TypeError as e:  # model passed missing/unexpected arguments
        result = f"Tool call failed: {e}"
    print(f"\n[tool] {name}({', '.join(f'{k}={v!r}' for k, v in args.items())}) -> {result}")
    return result


def run_turn(payload, speech, tools=None):
    """Run one user turn, executing tool calls until the model gives a final answer.

    Tool-call messages live only in this turn's payload. The caller stores just
    the final spoken text in recent_history, so rolling-summary memory only ever
    sees plain user/assistant text.
    """
    texts = []
    for _ in range(MAX_TOOL_ROUNDS):
        text, tool_calls = stream_reply(payload, speech, tools=tools)
        if text:
            texts.append(text)
        if not tool_calls or interrupted.is_set():
            break

        payload.append(
            {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    {"function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in tool_calls
                ],
            }
        )
        for call in tool_calls:
            payload.append({"role": "tool", "tool_name": call.function.name, "content": run_tool(call)})
        print("Bart > ", end="", flush=True)  # follow-up: Bart reports the result aloud
    print("\n")
    return " ".join(texts)


def strip_wake_word(text):
    """Drop a leading "max" / "hey max" from a transcript ("max open notepad" -> "open notepad")."""
    words = text.split()
    if words and words[0].lower() in GREETINGS and len(words) > 1 and words[1].lower() in WAKE_WORDS:
        return " ".join(words[2:])
    if words and words[0].lower() in WAKE_WORDS:
        return " ".join(words[1:])
    return text


def spoken_command(text):
    """Normalize a transcript for exit-word matching: lowercase, no punctuation, and
    without a trailing wake word ("shut down, max"), so those still match."""
    words = text.lower().strip(" .,!?").split()
    if len(words) > 1 and words[-1] in WAKE_WORDS:
        words.pop()
    return " ".join(words)


def set_state(ui, state):
    print(f"[state] {state}")
    ui.set_state(state)


def on_trigger(kind, speech):
    """Called from the microphone thread whenever a wake/stop phrase is heard."""
    print(f"[heard {kind!r}]")
    if turn_active.is_set():
        # Bart is thinking or talking: the wake word or "stop" cuts him off.
        if kind in ("max", "stop"):
            print(f"[interrupted by {kind!r}]")
            interrupted.set()
            speech.interrupt()
    elif kind == "max":
        wake_event.set()


def assistant_loop(ui, ears, speech, quit_app):
    """The whole voice session: wait for wake word -> listen -> think -> speak -> repeat."""
    running_summary = None
    recent_history = []

    while True:
        set_state(ui, IDLE)
        wake_event.clear()
        wake_event.wait()

        # Active session: keeps going while the user keeps talking within the window.
        while True:
            set_state(ui, LISTENING)
            interrupted.clear()
            text = ears.listen_for_command(first_word_timeout=ACTIVE_WINDOW_S)
            if not text:
                break  # 5 seconds of silence: back to idle (the widget fades out)
            text = strip_wake_word(text)
            if not text:
                continue  # only the wake word was heard (e.g. "max" said again): keep listening

            print(f"You > {text}")
            command = spoken_command(text)
            if command in QUIT_WORDS or command in END_WORDS:
                turn_active.set()
                speech.resume()
                set_state(ui, SPEAKING)
                print("Bart > Goodbye! Talk to you soon.")
                speech.say("Goodbye! Talk to you soon.")
                speech.wait()
                turn_active.clear()
                if command in QUIT_WORDS:
                    quit_app()
                    return
                break

            try:
                turn_active.set()
                speech.resume()
                set_state(ui, THINKING)
                recent_history.append({"role": "user", "content": text})
                print("Bart > ", end="", flush=True)
                reply = run_turn(
                    build_payload(running_summary, recent_history), speech, select_tools(text)
                )
                if reply:
                    recent_history.append({"role": "assistant", "content": reply})
                else:
                    recent_history.pop()  # nothing was said: keep user/assistant turns paired

                # Summarize while the last sentences are still being spoken.
                if len(recent_history) >= SUMMARY_TRIGGER:
                    running_summary = summarize(running_summary, recent_history)
                    recent_history = recent_history[-KEEP_AFTER_SUMMARY:]
                    print("[memory updated]\n")

                speech.wait()  # let all speech finish before listening again
            except Exception as e:  # e.g. Ollama not running; keep the assistant alive
                print(f"\n[error: {e}]")
                speech.resume()
                speech.say("Sorry, something went wrong.")
                speech.wait()
            finally:
                turn_active.clear()
            # Loop back: LISTENING for a follow-up (or straight away after an interruption).


def _setup_logging():
    """Under pythonw there is no console, so write output to bart.log instead."""
    if sys.stdout is None or sys.stderr is None:
        log = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = log


def main(source=None):
    """Run Bart. `source` replaces the microphone with fake audio (used for tests)."""
    _setup_logging()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # the orb hides itself; that must not end the app

    ui = OrbWidget()
    speech = SpeechQueue(on_speech_start=lambda: set_state(ui, SPEAKING))
    ears = Ears(lambda kind: on_trigger(kind, speech), source=source)

    def quit_app():  # safe to call from the assistant thread
        QMetaObject.invokeMethod(app, "quit", Qt.QueuedConnection)

    threading.Thread(
        target=assistant_loop, args=(ui, ears, speech, quit_app), daemon=True
    ).start()
    ears.start()

    # Let Ctrl+C work when run from a terminal.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    keepalive = QTimer()
    keepalive.start(250)
    keepalive.timeout.connect(lambda: None)

    print("Bart is ready. Say 'max' to wake me; say 'quit' to shut me down.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())  # an exception here reaches sys.excepthook, which logs it

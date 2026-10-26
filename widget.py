import math
import sys

from PySide6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QRadialGradient
from PySide6.QtWidgets import QApplication, QGraphicsOpacityEffect, QWidget

LISTENING, THINKING, SPEAKING, IDLE = "LISTENING", "THINKING", "SPEAKING", "IDLE"

STATE_COLORS = {
    LISTENING: QColor(0, 229, 255),   # cyan
    THINKING: QColor(167, 85, 255),   # purple
    SPEAKING: QColor(57, 255, 130),   # green
}
# (pulses per second, how far the orb swells) for each state
STATE_PULSE = {
    LISTENING: (0.7, 0.10),  # slow, calm breathing
    THINKING: (2.2, 0.08),   # quick, busy flutter
    SPEAKING: (3.0, 0.16),   # lively, voice-like
}

WINDOW_SIZE = 150
ORB_RADIUS = 30
SCREEN_MARGIN = 16
FADE_IN_MS = 300
FADE_OUT_MS = 2000


class OrbCanvas(QWidget):
    """Paints the glowing orb. Lives inside OrbWidget so the opacity effect can fade it."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.color = STATE_COLORS[LISTENING]
        self.state = LISTENING
        self.phase = 0.0

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)

        speed, swell = STATE_PULSE.get(self.state, STATE_PULSE[LISTENING])
        wave = math.sin(self.phase * speed * 2 * math.pi)
        if self.state == SPEAKING:  # a second, faster wave makes it feel like speech
            wave = 0.6 * wave + 0.4 * math.sin(self.phase * speed * 5.3 * 2 * math.pi)
        scale = 1.0 + swell * wave

        # Soft outer glow: the same color fading out to fully transparent.
        glow_r = ORB_RADIUS * 2.3 * scale
        glow = QRadialGradient(center, glow_r)
        for stop, alpha in ((0.0, 190), (0.35, 105), (0.65, 35), (1.0, 0)):
            c = QColor(self.color)
            c.setAlpha(alpha)
            glow.setColorAt(stop, c)
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(center, glow_r, glow_r)

        # Solid core with an off-center highlight so it reads as a sphere.
        core_r = ORB_RADIUS * scale
        core = QRadialGradient(center, core_r, center - QPointF(core_r * 0.3, core_r * 0.3))
        core.setColorAt(0.0, QColor(255, 255, 255, 235))
        core.setColorAt(0.45, self.color.lighter(125))
        core.setColorAt(1.0, self.color)
        p.setBrush(core)
        p.drawEllipse(center, core_r, core_r)


class OrbWidget(QWidget):
    """Frameless, transparent, always-on-top orb pinned to the top-right of the screen."""

    state_requested = Signal(str)  # lets any thread request a state change safely

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput  # clicks pass through to the app underneath
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)  # see-through window background
        self.setAttribute(Qt.WA_ShowWithoutActivating)  # never steal focus when appearing
        self.setFixedSize(WINDOW_SIZE, WINDOW_SIZE)

        self.canvas = OrbCanvas(self)
        self.canvas.setGeometry(0, 0, WINDOW_SIZE, WINDOW_SIZE)
        self.opacity = QGraphicsOpacityEffect(self.canvas)
        self.opacity.setOpacity(0.0)
        self.canvas.setGraphicsEffect(self.opacity)

        self.fade = QPropertyAnimation(self.opacity, b"opacity", self)
        self.fade.finished.connect(self._on_fade_finished)

        self.state = IDLE
        self.ticker = QTimer(self)  # drives the pulse animation (~60 fps)
        self.ticker.setInterval(16)
        self.ticker.timeout.connect(self._tick)

        self.state_requested.connect(self._apply_state)

    def set_state(self, state):
        """Thread-safe: may be called from any thread."""
        self.state_requested.emit(state)

    # --- internals (run on the Qt main thread) ---------------------------

    def _apply_state(self, state):
        self.state = state
        if state == IDLE:
            if self.isVisible():
                self._fade_to(0.0, FADE_OUT_MS)
            return
        self.canvas.state = state
        self.canvas.color = STATE_COLORS[state]
        if not self.isVisible():
            self._move_to_top_right()
            self.show()
        if self.opacity.opacity() < 1.0 or self.fade.state() == QPropertyAnimation.Running:
            self._fade_to(1.0, FADE_IN_MS)
        if not self.ticker.isActive():
            self.ticker.start()
        self.canvas.update()

    def _fade_to(self, target, duration_ms):
        self.fade.stop()
        self.fade.setStartValue(self.opacity.opacity())
        self.fade.setEndValue(target)
        self.fade.setDuration(duration_ms)
        self.fade.setEasingCurve(QEasingCurve.InOutQuad)
        self.fade.start()

    def _on_fade_finished(self):
        if self.state == IDLE and self.opacity.opacity() <= 0.001:
            self.ticker.stop()
            self.hide()

    def _tick(self):
        self.canvas.phase += 0.016
        self.canvas.update()

    def _move_to_top_right(self):
        area = QGuiApplication.primaryScreen().availableGeometry()
        self.move(area.right() - self.width() - SCREEN_MARGIN + 1, area.top() + SCREEN_MARGIN)


if __name__ == "__main__":
    # Demo: cycle through every state. Needs no microphone or models.
    app = QApplication(sys.argv)
    orb = OrbWidget()
    script = [(LISTENING, 3000), (THINKING, 3000), (SPEAKING, 3000), (LISTENING, 1500), (IDLE, 3000)]

    def run(i=0):
        if i >= len(script):
            app.quit()
            return
        state, ms = script[i]
        print(state)
        orb.set_state(state)
        QTimer.singleShot(ms, lambda: run(i + 1))

    run()
    sys.exit(app.exec())

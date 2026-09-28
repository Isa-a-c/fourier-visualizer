"""시간/차수 프레임을 재생하는 공통 Qt 창입니다."""
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QSpinBox, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class AnimationDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle('변화 과정 재생')
        self.resize(850, 600)
        layout = QVBoxLayout(self)
        figure = Figure(layout='constrained')
        self.canvas = FigureCanvasQTAgg(figure)
        axes = figure.subplots()
        axes.plot(data['x'], data['reference'], 'k--', label='Initial / original')
        self.line, = axes.plot(data['x'], data['frames'][0], label='Approximation')
        values = np.r_[np.ravel(data['frames']), data['reference']]
        low, high = float(values.min()), float(values.max())
        margin = max((high-low)*.08, 1e-6)
        axes.set(xlabel=data['xlabel'], ylabel=data['ylabel'], ylim=(low-margin, high+margin))
        axes.grid(alpha=.3)
        axes.legend()
        layout.addWidget(self.canvas)
        self.label = QLabel()
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(data['frames'])-1)
        layout.addWidget(self.slider)
        controls = QHBoxLayout()
        self.play = QPushButton('재생 / 일시정지')
        reset = QPushButton('처음으로')
        speed = QSpinBox()
        speed.setRange(50, 1000)
        speed.setValue(150)
        speed.setSuffix(' ms')
        for widget in (self.play, reset, speed):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(lambda: self.slider.setValue((self.slider.value()+1) % len(data['frames'])))
        speed.valueChanged.connect(self.timer.setInterval)
        self.play.clicked.connect(lambda: self.timer.stop() if self.timer.isActive() else self.timer.start())
        reset.clicked.connect(lambda: self.slider.setValue(0))
        def show_frame(index):
            self.line.set_ydata(data['frames'][index])
            self.label.setText(data['labels'][index])
            self.canvas.draw_idle()
        self.slider.valueChanged.connect(show_frame)
        show_frame(0)

    def done(self, result):
        self.timer.stop()
        super().done(result)

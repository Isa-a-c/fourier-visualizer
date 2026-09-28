"""주제의 입력 명세로 공통 폼을 생성합니다."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QFormLayout, QHBoxLayout, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit


class InputForm(QWidget):
    changed = Signal()

    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.fields = {}
        self.rows = {}
        self.form = QFormLayout(self)
        for spec in specs:
            if spec.kind == 'int':
                widget = QSpinBox()
                widget.setRange(int(spec.minimum), int(spec.maximum))
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(int(spec.minimum), int(spec.maximum))
                slider.valueChanged.connect(widget.setValue)
                widget.valueChanged.connect(slider.setValue)
                row = QWidget()
                layout = QHBoxLayout(row)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(slider)
                layout.addWidget(widget)
            elif spec.kind == 'float':
                widget = QDoubleSpinBox()
                widget.setDecimals(6)
                widget.setRange(spec.minimum, spec.maximum)
                row = widget
            elif spec.kind == 'choice':
                widget = QComboBox()
                widget.addItems(spec.choices)
                row = widget
            else:
                widget = QLineEdit()
                widget.setMaxLength(1000)
                row = widget
            widget.setToolTip(spec.help)
            self.fields[spec.key] = widget
            self.rows[spec.key] = row
            self.form.addRow(spec.label, row)
            if spec.kind in ('int', 'float'):
                widget.valueChanged.connect(lambda *_: self.changed.emit())
            elif spec.kind == 'choice':
                widget.currentTextChanged.connect(lambda *_: self.changed.emit())
            else:
                widget.textChanged.connect(lambda *_: self.changed.emit())
        self.set_values({})
        if 'mode' in self.fields:
            self.fields['mode'].currentTextChanged.connect(self.update_visibility)
        self.update_visibility()

    def values(self):
        values = {}
        for spec in self.specs:
            widget = self.fields[spec.key]
            values[spec.key] = widget.value() if spec.kind in ('int', 'float') else widget.currentText() if spec.kind == 'choice' else widget.text()
        return values

    def set_values(self, values):
        for spec in self.specs:
            widget, value = self.fields[spec.key], values.get(spec.key, spec.default)
            if spec.kind in ('int', 'float'):
                widget.setValue(value)
            elif spec.kind == 'choice':
                widget.setCurrentText(value)
            else:
                widget.setText(value)
        self.update_visibility()

    def update_visibility(self, *_):
        mode = self.fields['mode'].currentText() if 'mode' in self.fields else ''
        for spec in self.specs:
            self.form.setRowVisible(self.rows[spec.key], not spec.visible_in or mode in spec.visible_in)

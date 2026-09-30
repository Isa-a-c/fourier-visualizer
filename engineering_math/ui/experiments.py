"""지원되는 문제·영역 조합과 교재 바로가기를 제공하는 공통 선택기입니다."""
from PySide6.QtCore import Signal, QSignalBlocker
from PySide6.QtWidgets import QWidget, QFormLayout, QComboBox, QLabel
from engineering_math.topics.catalog import EXPERIMENTS, GROUPS, experiment_index


class ExperimentSelector(QWidget):
    currentIndexChanged = Signal(int)

    def __init__(self):
        super().__init__()
        self._index = -1
        self.group = QComboBox()
        self.group.addItems(GROUPS)
        self.problem = QComboBox()
        self.configuration = QComboBox()
        self.lesson = QComboBox()
        for i, item in enumerate(EXPERIMENTS):
            self.lesson.addItem(item.lesson, i)
        self.method = QLabel()
        self.method.setWordWrap(True)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for label, widget in [('실험군', self.group), ('문제', self.problem),
                              ('영역·구성', self.configuration), ('해법', self.method),
                              ('교재 바로가기', self.lesson)]:
            layout.addRow(label, widget)
        self.group.currentTextChanged.connect(self.choose_group)
        self.problem.currentTextChanged.connect(self.choose_problem)
        self.configuration.currentIndexChanged.connect(self.choose_configuration)
        self.lesson.activated.connect(lambda _: self.setCurrentIndex(self.lesson.currentData()))
        self.setCurrentIndex(0)

    @property
    def experiment(self):
        return EXPERIMENTS[self._index]

    def currentData(self):
        return self.experiment.topic

    def currentIndex(self):
        return self._index

    def findData(self, topic):
        return experiment_index(topic)

    def select(self, topic, params=None):
        self.setCurrentIndex(experiment_index(topic, params))

    def setCurrentIndex(self, index):
        if index == self._index:
            return
        if not 0 <= index < len(EXPERIMENTS):
            raise ValueError('지원하지 않는 실험 구성입니다.')
        entry = EXPERIMENTS[index]
        blockers = [QSignalBlocker(widget) for widget in (self.group, self.problem, self.configuration, self.lesson)]
        self.group.setCurrentText(entry.group)
        self.problem.clear()
        self.problem.addItems(list(dict.fromkeys(item.problem for item in EXPERIMENTS if item.group == entry.group)))
        self.problem.setCurrentText(entry.problem)
        self.configuration.clear()
        for i, item in enumerate(EXPERIMENTS):
            if item.group == entry.group and item.problem == entry.problem:
                self.configuration.addItem(item.configuration, i)
        self.configuration.setCurrentIndex(self.configuration.findData(index))
        self.lesson.setCurrentIndex(index)
        self.method.setText(entry.method)
        self._index = index
        del blockers
        self.currentIndexChanged.emit(index)

    def choose_group(self, group):
        self.setCurrentIndex(next(i for i, item in enumerate(EXPERIMENTS) if item.group == group))

    def choose_problem(self, problem):
        self.setCurrentIndex(next(i for i, item in enumerate(EXPERIMENTS)
                                  if item.group == self.group.currentText() and item.problem == problem))

    def choose_configuration(self, _):
        index = self.configuration.currentData()
        if index is not None:
            self.setCurrentIndex(index)

"""주제 교체에 독립적인 공업수학 학습 화면입니다."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QComboBox, QLabel, QPushButton, QScrollArea, QTabWidget, QProgressBar, QMessageBox,
    QFileDialog, QTableView, QTextBrowser)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from engineering_math import VERSION
from engineering_math.topics import get_topics
from engineering_math.topics.lessons import lesson_for
from engineering_math.core.project import save_settings, load_settings, export_results
from engineering_math.core.models import validate_params
from .forms import InputForm
from .jobs import CalculationJob
from .animation import AnimationDialog


class ResultTable(QAbstractTableModel):
    """표의 셀마다 위젯을 생성하지 않고 필요한 값만 표시합니다."""
    def __init__(self, columns, rows, parent=None):
        super().__init__(parent)
        self.columns, self.rows = columns, rows

    def rowCount(self, parent=None):
        return len(self.rows)

    def columnCount(self, parent=None):
        return len(self.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and index.isValid():
            value = self.rows[index.row()][index.column()]
            return f'{value:.8g}' if isinstance(value, (float, np.floating)) else str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self.columns[section] if orientation == Qt.Orientation.Horizontal else str(section+1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'공업수학 학습 스튜디오 {VERSION}')
        self.resize(1400, 900)
        self.topics = get_topics()
        self.states = {}
        self.topic = None
        self.form = None
        self.result = None
        self.figures = []
        self.job = CalculationJob(self)
        self.job.completed.connect(self.present)
        self.job.failed.connect(self.show_error)

        # 공통 입력 패널: 새 주제는 명세만 제공하면 기본 폼을 사용할 수 있습니다.
        splitter = QSplitter()
        self.setCentralWidget(splitter)
        left = QWidget()
        layout = QVBoxLayout(left)
        self.selector = QComboBox()
        for topic in self.topics.values():
            self.selector.addItem(topic.title, topic.id)
        layout.addWidget(self.selector)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.examples = QComboBox()
        layout.addWidget(self.examples)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)
        controls = QHBoxLayout()
        calculate = QPushButton('계산하기')
        cancel = QPushButton('계산 취소')
        reset = QPushButton('초기화')
        for widget in (calculate, cancel, reset):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.animate = QPushButton('변화 과정 재생')
        self.animate.setEnabled(False)
        layout.addWidget(self.animate)
        self.laplace_link = QPushButton('이 ODE 결과를 라플라스 역변환으로 확인')
        self.laplace_link.setEnabled(False)
        self.laplace_link.clicked.connect(self.open_laplace)
        layout.addWidget(self.laplace_link)
        splitter.addWidget(left)

        right = QWidget()
        output = QVBoxLayout(right)
        self.metrics = QLabel('입력값을 확인한 후 계산하기를 누르십시오.')
        self.metrics.setWordWrap(True)
        output.addWidget(self.metrics)
        self.input_state = QLabel()
        self.input_state.setWordWrap(True)
        output.addWidget(self.input_state)
        self.tabs = QTabWidget()
        output.addWidget(self.tabs)
        self.notices = QLabel()
        self.notices.setWordWrap(True)
        self.notices.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        output.addWidget(self.notices)
        splitter.addWidget(right)
        splitter.setSizes([430, 970])
        self.selector.currentIndexChanged.connect(self.change_topic)
        self.examples.activated.connect(self.apply_example)
        calculate.clicked.connect(self.calculate)
        cancel.clicked.connect(self.cancel)
        reset.clicked.connect(lambda: self.form.set_values({}))
        self.animate.clicked.connect(self.show_animation)

        menu = self.menuBar().addMenu('파일')
        menu.addAction('설정 저장…', self.save)
        menu.addAction('설정 불러오기…', self.load)
        menu.addAction('결과 CSV 묶음 저장…', self.export)
        menu.addSeparator()
        menu.addAction('종료', self.close)
        self.menuBar().addAction('학습 안내', self.help)
        self.change_topic()

    def clear_output(self):
        import matplotlib.pyplot as plt
        self.result = None
        self.animate.setEnabled(False)
        self.laplace_link.setEnabled(False)
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()
        for figure in self.figures:
            plt.close(figure)
        self.figures.clear()

    def change_topic(self, *_):
        self.cancel()
        if self.topic is not None:
            self.states[self.topic.id] = self.form.values()
        self.topic = self.topics[self.selector.currentData()]
        self.laplace_link.setVisible(self.topic.id == 'ode')
        self.description.setText(self.topic.description)
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.form = InputForm(self.topic.inputs)
        self.form.set_values(self.states.get(self.topic.id, {}))
        self.form.changed.connect(self.update_input_state)
        self.scroll.setWidget(self.form)
        self.examples.clear()
        self.examples.addItem('예제를 선택하십시오')
        self.examples.addItems(list(self.topic.examples))
        self.clear_output()
        self.add_lesson()
        self.update_input_state()
        self.metrics.setText('입력값을 확인한 후 계산하기를 누르십시오.')
        self.notices.clear()

    def apply_example(self, index):
        if index > 0:
            self.form.set_values(self.topic.examples[self.examples.itemText(index)])

    def add_lesson(self):
        browser = QTextBrowser()
        browser.setMarkdown(lesson_for(self.topic.id))
        self.tabs.addTab(browser, '학습 설명')

    def update_input_state(self):
        try:
            current = validate_params(self.topic.inputs, self.form.values())
        except ValueError:
            current = None
        if self.result is None:
            self.input_state.setText('아직 계산 결과가 없습니다.')
            self.input_state.setStyleSheet('')
        elif current != self.result.params:
            self.input_state.setText('입력값이 변경되었습니다. 다시 계산하십시오. 표시·저장되는 결과는 이전 계산의 결과입니다.')
            self.input_state.setStyleSheet('color: #9a4d00; font-weight: bold;')
        else:
            self.input_state.setText('현재 입력값과 계산 결과가 일치합니다.')
            self.input_state.setStyleSheet('')

    def calculate(self):
        self.progress.show()
        self.statusBar().showMessage('계산 중입니다. 취소하거나 다른 주제로 이동할 수 있습니다.')
        try:
            self.job.start(self.topic.id, self.form.values())
        except Exception as exc:
            self.show_error(str(exc))

    def cancel(self):
        self.job.cancel()
        self.progress.hide()
        self.statusBar().showMessage('계산이 중지되었습니다.')

    def present(self, result):
        self.progress.hide()
        if result.topic != self.topic.id:
            return
        try:
            self.clear_output()
            figures = self.topic.figures(result)
            for name, figure in figures.items():
                page = QWidget()
                layout = QVBoxLayout(page)
                canvas = FigureCanvasQTAgg(figure)
                layout.addWidget(NavigationToolbar2QT(canvas, page))
                layout.addWidget(canvas)
                self.tabs.addTab(page, name)
                self.figures.append(figure)
                canvas.draw()
            table_tabs = QTabWidget()
            for name, (columns, rows) in result.tables.items():
                table = QTableView()
                table.setModel(ResultTable(columns, rows, table))
                table.setAlternatingRowColors(True)
                table_tabs.addTab(table, name)
            self.tabs.addTab(table_tabs, '수치 표')
            if 'symbolic' in result.data:
                symbolic_view = QTextBrowser()
                symbolic_view.setPlainText(result.data['symbolic'])
                self.tabs.addTab(symbolic_view, '기호 계산')
            settings_view = QTextBrowser()
            lines = ['계산 당시 입력값', '']
            for spec in self.topic.inputs:
                lines.append(f'{spec.label}: {result.params[spec.key]}')
            settings_view.setPlainText('\n'.join(lines))
            self.tabs.addTab(settings_view, '계산 설정')
            self.add_lesson()
            self.metrics.setText('   |   '.join(f'{name}: {value:.6g}' for name, value in result.metrics.items()))
            self.notices.setText('\n'.join(result.notices))
            self.result = result
            self.update_input_state()
            self.animate.setEnabled(getattr(self.topic, 'supports_animation', True))
            self.laplace_link.setEnabled(result.topic == 'ode' and bool(result.data.get('laplace_input')))
            self.statusBar().showMessage('계산이 완료되었습니다. 결과는 계산 당시 입력값을 기준으로 합니다.')
        except Exception as exc:
            self.show_error(str(exc))

    def show_error(self, message):
        self.progress.hide()
        self.statusBar().showMessage('계산 또는 파일 작업에 실패했습니다.')
        QMessageBox.warning(self, '입력 및 계산 확인', message)

    def show_animation(self):
        if self.result is not None:
            try:
                dialog = AnimationDialog(self.topic.animation(self.result), self)
                dialog.exec()
            except Exception as exc:
                self.show_error(str(exc))

    def open_laplace(self):
        if self.result is None or self.result.topic != 'ode' or not self.result.data.get('laplace_input'):
            return
        params = {'mode': '역변환 s → t', 'expression': self.result.data['laplace_input'],
                  'end_time': self.result.params['end_time'], 'num_points': self.result.params['num_points']}
        self.selector.setCurrentIndex(self.selector.findData('laplace'))
        self.form.set_values(params)
        self.calculate()

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, '설정 저장', '', 'JSON (*.json)')
        if path:
            try:
                save_settings(path, self.topic.id, self.form.values())
            except Exception as exc:
                self.show_error(str(exc))

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, '설정 불러오기', '', 'JSON (*.json)')
        if path:
            try:
                self.load_path(path)
            except Exception as exc:
                self.show_error(str(exc))

    def load_path(self, path):
        """대화상자와 분리하여 실제 설정 로딩 경로를 검사할 수 있습니다."""
        data = load_settings(path)
        self.selector.setCurrentIndex(self.selector.findData(data['topic']))
        self.form.set_values(data['params'])
        self.calculate()

    def export(self):
        if self.result is None:
            self.show_error('먼저 계산을 완료하십시오.')
            return
        path, _ = QFileDialog.getSaveFileName(self, '결과 저장', '', 'ZIP (*.zip)')
        if path:
            try:
                export_results(path, self.result)
                self.statusBar().showMessage(f'저장되었습니다: {Path(path).name}')
            except Exception as exc:
                self.show_error(str(exc))

    def help(self):
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == '학습 설명':
                self.tabs.setCurrentIndex(index)
                break

    def closeEvent(self, event):
        self.cancel()
        self.clear_output()
        event.accept()

"""같은 PC에서 변경 전후를 비교하는 공통 화면 벤치마크입니다."""
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(label):
    from PySide6.QtWidgets import QApplication
    from engineering_math.ui.window import MainWindow
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    errors = []
    window.show_error = errors.append
    rows = []
    for topic_id in ('fourier', 'heat', 'disk_membrane'):
        window.selector.setCurrentIndex(window.selector.findData(topic_id))
        topic = window.topic
        result = topic.compute(window.form.values())
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            window.present(result)
            application.processEvents()
            timings.append((time.perf_counter() - start) * 1000)
        assert not errors, errors
        rows.append({'topic': topic_id, 'present_ms': median(timings),
                     'table_array_bytes': sum(getattr(rows, 'nbytes', 0) for _, rows in result.tables.values())})
    window.close()
    cold = []
    snippet = 'from engineering_math.topics import get_topic; get_topic("fourier")'
    for _ in range(3):
        start = time.perf_counter()
        subprocess.run([sys.executable, '-c', snippet], cwd=ROOT, check=True)
        cold.append((time.perf_counter() - start) * 1000)
    report = {'label': label, 'cold_fourier_ms': median(cold), 'results': rows,
              'scope': 'Same PC; median of 3; present includes UI events, excludes compute; cold includes Python startup; table bytes exclude shared source arrays'}
    target = ROOT / 'build' / f'workbench-{label}.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'current')

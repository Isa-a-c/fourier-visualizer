"""수식 평가와 계산을 종료 가능한 별도 프로세스에서 수행합니다."""
import multiprocessing as mp
import queue
from PySide6.QtCore import QObject, QTimer, Signal


def compute_worker(output, topic_id, params, cache):
    try:
        from engineering_math.topics import get_topic
        topic = get_topic(topic_id)
        if hasattr(topic, 'restore_cache'):
            topic.restore_cache(cache)
        result = topic.compute(params)
        snapshot = topic.snapshot_cache() if hasattr(topic, 'snapshot_cache') else None
        output.put((True, result, snapshot))
    except Exception as exc:
        output.put((False, f'{type(exc).__name__}: {exc}', None))


class CalculationJob(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.output = None
        # 마지막 성공 결과의 캐시만 보관합니다. 실패·취소는 캐시를 교체하지 않습니다.
        self.caches = {}
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.poll)

    def start(self, topic_id, params):
        self.cancel()
        context = mp.get_context('spawn')
        self.output = context.Queue()
        self.process = context.Process(target=compute_worker, args=(self.output, topic_id, params, self.caches.get(topic_id)), daemon=True)
        self.process.start()
        self.timer.start()

    def poll(self):
        try:
            success, payload, snapshot = self.output.get_nowait()
        except queue.Empty:
            if self.process is not None and not self.process.is_alive():
                self.cancel()
                self.failed.emit('계산 프로세스가 결과 없이 종료되었습니다.')
            return
        self.cancel()
        if success:
            if snapshot is not None:
                self.caches[payload.topic] = snapshot
            self.completed.emit(payload)
        else:
            self.failed.emit(payload)

    def cancel(self):
        self.timer.stop()
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
            self.process.close()
            self.process = None
        if self.output is not None:
            self.output.close()
            self.output = None

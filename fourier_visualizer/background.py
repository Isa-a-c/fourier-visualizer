"""Tk와 계산 프로세스를 분리한다. 취소/새 요청 시 이전 프로세스를 종료한다."""
import multiprocessing as mp
from queue import Empty


def calculate_request(kind, payload):
    """작업 프로세스에서는 Tk나 Matplotlib Figure를 만들지 않는다."""
    if kind == 'series':
        import numpy as np
        from fourier_core import FourierSession, parse_expression, calculate_fourier_sums, calculate_errors
        settings, cache = payload
        L = float(parse_expression(settings['L']))
        if not np.isfinite(L) or L <= 0 or not np.isfinite(2*L):
            raise ValueError('L과 주기는 유한한 양수여야 합니다.')
        session = FourierSession()
        session.key, session.data = cache
        N, selected = settings['N'], settings['comparison_N']
        expression,x,y,a0,an,bn = session.prepare(settings['function'],L,settings['num_points'],max([N,*selected]))
        sums = calculate_fourier_sums(x,L,a0,an,bn,[N,*selected])
        error, metrics = calculate_errors(y,sums[N])
        convergence = []
        for n in selected:
            values = metrics if n == N else calculate_errors(y,sums[n])[1]
            convergence.append([n,values['MSE'],values['RMSE']])
        return (expression,x,y,L,N,selected,a0,an,bn,sums,error,metrics,convergence,(session.key,session.data))
    if kind == 'analysis':
        from advanced import prepare_analysis
        mode,data,options = payload
        return prepare_analysis(mode,data,**options)
    if kind == 'animation':
        from fourier_core import calculate_fourier_coefficients
        _,x,y,L,*_ = payload
        return calculate_fourier_coefficients(x,y,L,100)
    raise ValueError('지원하지 않는 작업입니다.')


def worker(queue, kind, payload):
    try:
        queue.put((True,calculate_request(kind,payload)))
    except Exception as exc:
        queue.put((False,str(exc)))


class BackgroundJob:
    def __init__(self, widget, state_callback):
        self.widget = widget
        self.state_callback = state_callback
        self.process = None
        self.queue = None
        self.poll_id = None
        self.generation = 0

    @property
    def busy(self):
        return self.process is not None

    def start(self, kind, payload, success, failure):
        self.cancel(notify=False)
        generation = self.generation
        context = mp.get_context('spawn')
        self.queue = context.Queue()
        self.process = context.Process(target=worker,args=(self.queue,kind,payload),daemon=True)
        self.success, self.failure = success, failure
        try:
            self.process.start()
        except Exception as exc:
            self.process = None
            self.queue.close()
            self.queue = None
            failure(str(exc))
            return
        self.state_callback(True,'계산 중입니다. 취소하실 수 있습니다.')
        self.poll_id = self.widget.after(40,lambda:self.poll(generation))

    def poll(self, generation):
        self.poll_id = None
        if generation != self.generation or self.process is None: return
        try:
            ok,result = self.queue.get_nowait()
        except Empty:
            if not self.process.is_alive():
                # 생산 프로세스 종료 후 큐 전달을 마지막으로 확인한다.
                try: ok,result = self.queue.get(timeout=.02)
                except Empty: ok,result = False,'계산 프로세스가 결과 없이 종료되었습니다.'
            else:
                self.poll_id = self.widget.after(40,lambda:self.poll(generation))
                return
        success,failure = self.success,self.failure
        self.cancel(notify=False)
        self.state_callback(False,'계산 완료' if ok else '계산 오류')
        if ok:
            try: success(result)
            except Exception as exc: failure(str(exc))
        else: failure(result)

    def cancel(self, notify=True):
        self.generation += 1
        if self.poll_id is not None:
            self.widget.after_cancel(self.poll_id)
            self.poll_id = None
        if self.process is not None:
            if self.process.is_alive(): self.process.terminate()
            self.process.join(timeout=.2)
            if not self.process.is_alive(): self.process.close()
            self.process = None
        if self.queue is not None:
            self.queue.close()
            self.queue = None
        if notify: self.state_callback(False,'취소되었습니다. 이전 결과는 마지막으로 완료된 계산입니다.')

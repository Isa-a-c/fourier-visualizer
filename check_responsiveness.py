"""실제 spawn 프로세스와 Tk 이벤트 루프의 취소·완료·종료 회귀 검사."""
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'fourier_visualizer'))


def run():
    import numpy as np
    from desktop_app import FourierDesktop
    window=FourierDesktop()
    window.withdraw()
    window.recalculate()
    callbacks=[]
    window.after(10,lambda:callbacks.append('heartbeat'))
    window.order.set(100)
    window.points.set(20000)
    window.request_calculation()
    assert window.calculate_job.busy
    deadline=time.monotonic()+30
    while window.calculate_job.busy and time.monotonic()<deadline:
        window.update()
        time.sleep(.01)
    assert not window.calculate_job.busy,'worker timeout'
    assert callbacks==['heartbeat'] and not window.status.get(),window.status.get()
    assert window.last_settings['N']==100 and len(window.last_result[0])==20000
    window.expression.set('x**2')
    window.request_calculation()
    cancelled_process=window.calculate_job.process
    window.cancel_calculation()
    assert not window.calculate_job.busy
    window.expression.set('sin(x)')
    window.request_calculation()
    deadline=time.monotonic()+30
    while window.calculate_job.busy and time.monotonic()<deadline:
        window.update(); time.sleep(.01)
    assert window.last_settings['function']=='sin(x)' and not window.status.get()
    panel=window.advanced_panel
    panel.mode.set('적분 안정성')
    panel.request_render()
    deadline=time.monotonic()+30
    while panel.job.busy and time.monotonic()<deadline:
        window.update(); time.sleep(.01)
    assert panel.figure is not None,panel.note.get()
    animation=window.open_animation()
    animation.withdraw()
    animation.axes.set_ylim(-.1,.1)
    animation.auto_y.set(False)
    animation.values[:]=2
    animation.update_limits()
    assert '주의' in animation.range_note.get()
    animation.auto_y.set(True)
    animation.update_limits()
    low,high=animation.axes.get_ylim()
    assert low <= np.min(animation.original_y) and high >= 2
    panel.request_render()
    window.request_calculation()
    window.close()
    assert not panel.job.busy and not window.calculate_job.busy
    print('PASS: live UI heartbeat, async completion, cancellation, newer result, async analysis, axis clipping, close during work')


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    run()

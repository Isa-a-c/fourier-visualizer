"""패키징 진입점과 배포본의 재현 가능한 진단 실행."""
import sys
import json
from pathlib import Path


def main():
    from desktop_app import FourierDesktop
    window = FourierDesktop()
    if len(sys.argv)==3 and sys.argv[1]=='--self-test':
        window.withdraw()
        try:
            window.update()
            window.recalculate()
            if window.status.get(): raise RuntimeError(window.status.get())
            import time
            heartbeat=[]
            window.after(10,lambda:heartbeat.append(True))
            window.order.set(100)
            window.request_calculation()
            deadline=time.monotonic()+40
            while window.calculate_job.busy and time.monotonic()<deadline:
                window.update()
                time.sleep(.01)
            if window.calculate_job.busy or not heartbeat or window.status.get():
                raise RuntimeError('백그라운드 계산 또는 UI heartbeat 실패')
            window.request_calculation()
            window.cancel_calculation()
            panel = window.advanced_panel
            for mode in ['Gibbs 확대','진폭·위상','DFT·FFT·급수 비교','Aliasing 실험','적분 안정성','N별 오차 곡선']:
                panel.mode.set(mode)
                panel.render()
                if panel.figure is None: raise RuntimeError(panel.note.get())
            panel.mode.set('적분 안정성')
            panel.request_render()
            deadline=time.monotonic()+40
            while panel.job.busy and time.monotonic()<deadline:
                window.update(); time.sleep(.01)
            if panel.job.busy or panel.figure is None: raise RuntimeError('확장 비동기 계산 실패')
            animation=window.open_animation()
            animation.withdraw()
            animation.axes.set_ylim(-.1,.1)
            animation.values.fill(2)
            animation.update_limits()
            if animation.axes.get_ylim()[1]<2: raise RuntimeError('애니메이션 축 확장 실패')
            animation.reset(); animation.play(); animation.pause(); animation.close()
            import tempfile
            import zipfile
            from project_io import export_results,export_transform
            from fourier_core import parse_expression
            with tempfile.TemporaryDirectory(prefix='푸리에 검증 ') as directory:
                result_path=Path(directory)/'급수 결과.zip'
                dft_path=Path(directory)/'변환 결과.zip'
                export_results(result_path,window.last_settings,window.last_result)
                export_transform(dft_path,(parse_expression('sin(x)'),3.141592653589793,64))
                for archive_path in [result_path,dft_path]:
                    with zipfile.ZipFile(archive_path) as archive:
                        if archive.testzip() is not None: raise RuntimeError('내보내기 검증 실패')
            Path(sys.argv[2]).write_text(json.dumps(dict(ok=True,frozen=bool(getattr(sys,'frozen',False)),
                executable=sys.executable,modes=6,background=True,cancel=True,animation=True,
                unicode_export=True),indent=2),encoding='utf-8')
        except Exception as exc:
            Path(sys.argv[2]).write_text(json.dumps(dict(ok=False,error=str(exc))),encoding='utf-8')
            raise
        finally:
            window.close()
    else:
        window.mainloop()


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

"""소스와 실행 파일에서 같은 Qt 통합 검사를 수행합니다."""
import json
import time
import tempfile
from pathlib import Path
from engineering_math.topics.fourier import MODES
from engineering_math.core.project import save_settings, load_settings, export_results
from engineering_math.ui.animation import AnimationDialog


def wait_for_job(application, window):
    deadline = time.monotonic()+90
    pulses = 0
    while window.job.process is not None and time.monotonic() < deadline:
        application.processEvents()
        pulses += 1
        time.sleep(.01)
    assert window.job.process is None, '계산 완료 대기 시간이 초과되었습니다.'
    return pulses


def run(application, window, report_path):
    checks = []
    errors = []
    window.show_error = errors.append
    window.job.failed.disconnect()
    window.job.failed.connect(errors.append)
    try:
        window.show()
        application.processEvents()
        # 실제 Qt 이벤트 루프를 펌프하며 별도 계산 프로세스의 완료를 기다립니다.
        window.calculate()
        pulses = wait_for_job(application, window)
        assert window.result is not None and not errors, errors
        assert pulses > 2
        checks.append('Qt 비동기 푸리에 계산 및 화면 응답')
        for key, value in [('N', 37), ('num_points', 7311)]:
            window.form.fields[key].setValue(value)
            from PySide6.QtWidgets import QSlider
            slider = window.form.rows[key].findChild(QSlider)
            assert slider.value() == value
            slider.setValue(value+1)
            assert window.form.fields[key].value() == value+1
        checks.append('N 및 표본 수 숫자 입력·슬라이더 양방향 동기화')
        assert '변경' in window.input_state.text()
        window.form.set_values({'N': 20})
        window.calculate()
        # 계산 중 수정한 입력을 완료된 결과와 혼동하지 않아야 합니다.
        window.form.fields['N'].setValue(21)
        wait_for_job(application, window)
        assert window.result.params['N'] == 20
        assert window.result.data['cache_reused'] == 10
        assert window.result.data['cache_integrated'] == 10
        assert '변경' in window.input_state.text()
        window.form.fields['N'].setValue(20)
        assert '일치' in window.input_state.text()
        checks.append('프로세스 사이 계수 재사용 및 계산 중 입력 변경 안내')
        window.form.fields['function'].setText('1/0')
        window.calculate()
        wait_for_job(application, window)
        assert errors
        errors.clear()
        window.form.set_values({'N': 20})
        window.calculate()
        wait_for_job(application, window)
        assert not errors and window.result.data['cache_integrated'] == 0
        checks.append('잘못된 입력 이후 복구 및 성공 캐시 보존')
        for mode in MODES:
            result = window.topic.compute({'mode': mode})
            window.present(result)
            assert window.result is result and not errors, errors
            application.processEvents()
        checks.append('푸리에 기본 및 추가 분석 6종 그래프 표시')
        dialog = AnimationDialog(window.topic.animation(window.result), window)
        dialog.slider.setValue(dialog.slider.maximum())
        dialog.close()
        checks.append('푸리에 부분합 애니메이션')
        window.calculate()
        window.cancel()
        assert window.job.process is None
        checks.append('계산 취소')
        window.selector.setCurrentIndex(window.selector.findData('heat'))
        result = window.topic.compute({})
        window.present(result)
        assert window.tabs.count() == 6 and not errors, errors
        checks.append('열방정식 곡선·히트맵·수치 표 표시')
        application.processEvents()
        screenshot = Path(report_path).with_suffix('.png')
        window.grab().save(str(screenshot))
        dialog = AnimationDialog(window.topic.animation(result), window)
        dialog.show()
        dialog.slider.setValue(dialog.slider.maximum())
        application.processEvents()
        dialog.close()
        checks.append('열방정식 애니메이션 마지막 프레임')
        window.calculate()
        window.selector.setCurrentIndex(window.selector.findData('wave'))
        assert window.job.process is None and window.result is None
        window.calculate()
        wait_for_job(application, window)
        result = window.result
        assert result.topic == 'wave' and window.tabs.count() == 7 and not errors, errors
        checks.append('계산 중 주제 변경 및 파동 비동기 계산·열 비교 그래프')
        window.resize(1000, 700)
        application.processEvents()
        window.grab().save(str(Path(report_path).with_name('wave-small.png')))
        window.help()
        assert window.tabs.tabText(window.tabs.currentIndex()) == '학습 설명'
        assert '초기 속도' in window.tabs.currentWidget().toPlainText()
        dialog = AnimationDialog(window.topic.animation(result), window)
        dialog.slider.setValue(dialog.slider.maximum())
        dialog.close()
        checks.append('파동 애니메이션·학습 설명 및 작은 창 표시')
        with tempfile.TemporaryDirectory(prefix='공업수학 검증 ') as folder:
            path = Path(folder)/'설정.json'
            save_settings(path, result.topic, result.params)
            assert load_settings(path)['topic'] == 'wave'
            export_results(Path(folder)/'결과.zip', result)
            window.selector.setCurrentIndex(window.selector.findData('fourier'))
            window.load_path(path)
            wait_for_job(application, window)
            assert window.result.topic == 'wave' and not errors
        checks.append('한글 경로 설정 및 결과 저장')
        from engineering_math.core.expr import parse_expression, evaluate
        assert evaluate(parse_expression('besselj(0,x)'), ([0.],))[0] == 1.
        checks.append('SciPy 특수함수 평가')
        window.selector.setCurrentIndex(window.selector.findData('ode'))
        window.calculate()
        wait_for_job(application, window)
        assert window.result.topic == 'ode' and not errors, errors
        assert window.laplace_link.isEnabled() and not window.animate.isEnabled()
        application.processEvents()
        window.grab().save(str(Path(report_path).with_name('ode-window.png')))
        window.open_laplace()
        wait_for_job(application, window)
        assert window.result.topic == 'laplace' and 'values' in window.result.data and not errors, errors
        checks.append('ODE 시간 응답·위상평면 및 초기조건 포함 라플라스 역변환 연결')
        for index in range(window.tabs.count()):
            if window.tabs.tabText(index) == '기호 계산':
                window.tabs.setCurrentIndex(index)
        application.processEvents()
        window.grab().save(str(Path(report_path).with_name('laplace-window.png')))
        window.form.set_values({'mode': '역변환 s → t', 'expression': '1'})
        window.calculate()
        wait_for_job(application, window)
        assert 'values' not in window.result.data and not errors, errors
        checks.append('임펄스 역변환의 기호 표시 및 그래프 생략')
        window.selector.setCurrentIndex(window.selector.findData('ode'))
        window.form.set_values({'mode': '직접 입력', 'force_expression': 'A*cos(w*t)',
                                'parameters': 'A=2; w=3', 'end_time': 2., 'num_points': 201})
        window.calculate()
        wait_for_job(application, window)
        assert not errors and window.result.data['laplace_input'] is None, errors
        assert not window.laplace_link.isEnabled()
        assert window.result.params['parameters'] == 'A=2; w=3'
        application.processEvents()
        window.grab().save(str(Path(report_path).with_name('custom-force.png')))
        window.form.fields['symbolic_mode'].setCurrentText('라플라스 변환도 계산')
        window.calculate()
        wait_for_job(application, window)
        assert not errors and window.laplace_link.isEnabled(), errors
        window.open_laplace()
        wait_for_job(application, window)
        assert not errors and 'values' in window.result.data, errors
        checks.append('매개변수 외력 직접 입력·수치 계산·선택적 라플라스 연결')
        window.selector.setCurrentIndex(window.selector.findData('fourier'))
        window.form.set_values({'function': 'A*sinc(x)', 'parameters': 'A=2'})
        window.calculate()
        wait_for_job(application, window)
        assert not errors and window.result.params['parameters'] == 'A=2', errors
        window.help()
        assert '매개변수' in window.tabs.currentWidget().toPlainText()
        checks.append('확장 함수 및 공통 매개변수 입력·도움말')
        window.calculate()
        window.close()
        assert window.job.process is None
        checks.append('계산 중 창 종료 시 작업 프로세스 정리')
        import sys
        report = dict(passed=True, frozen=bool(getattr(sys, 'frozen', False)), checks=checks)
        code = 0
    except Exception as exc:
        import traceback
        report = dict(passed=False, checks=checks, error=str(exc), traceback=traceback.format_exc())
        code = 1
    finally:
        window.close()
        application.processEvents()
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return code

"""단측 라플라스 변환과 인과적 역변환의 기호 학습 모듈입니다."""
import numpy as np
import sympy as sp
from engineering_math.core.expr import parameter_expression, evaluate, symbols_for
from engineering_math.core.models import InputSpec, Result, validate_params

MODES = ('정변환 t → s', '역변환 s → t')


class LaplaceTopic:
    id = 'laplace'
    title = '라플라스 변환 · 역변환'
    description = '단측 라플라스 변환식과 수렴 조건을 확인합니다. 역변환은 인과적 해석을 사용합니다.'
    supports_animation = False
    inputs = (
        InputSpec('mode', '계산 방향', 'choice', MODES[0], choices=MODES),
        InputSpec('expression', '수식 (t 또는 s)', 'text', 'sin(2*t)'),
        InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
        InputSpec('end_time', '그래프 마지막 시간', 'float', 10., .01, 100),
        InputSpec('num_points', '그래프 표본 수', 'int', 1001, 101, 5001),
    )
    examples = {'사인 정변환': {}, '지수 감쇠': {'expression': 'exp(-2*t)'},
                '지연 계단': {'expression': 'Heaviside(t-2)'},
                '이차식 역변환': {'mode': MODES[1], 'expression': '1/(s**2+4)'},
                '중근 역변환': {'mode': MODES[1], 'expression': '1/(s+1)**2'},
                '임펄스 역변환': {'mode': MODES[1], 'expression': '1'},
                '매개변수 정변환': {'expression': 'A*exp(-w*t)', 'parameters': 'A=2; w=3'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        t = symbols_for(('t',))[0]
        s = sp.Symbol('s')
        notices = ['단측 변환을 사용하며 s는 복소수 변수입니다. 시간 그래프는 t>0의 표본입니다.']
        if p['mode'] == MODES[0]:
            time_expression = parameter_expression(p['expression'], p['parameters'], ('t',))
            try:
                transformed, plane, condition = sp.laplace_transform(time_expression, t, s)
            except Exception as exc:
                transformed, plane, condition = sp.LaplaceTransform(time_expression, t, s), None, None
                notices.append(f'기호 변환을 완료하지 못했습니다({type(exc).__name__}). 원함수의 수치 그래프는 별도로 시도합니다.')
            rows = [['f(t)', str(time_expression)], ['F(s)', str(transformed)],
                    ['수렴 영역', f'Re(s) > {plane}'], ['추가 조건', str(condition)]]
            if transformed.has(sp.LaplaceTransform):
                rows[2] = ['수렴 영역', '미확정: 변환이 미평가 상태입니다.']
                notices.append('변환 일부가 닫힌 형태로 계산되지 않았습니다. 미평가 변환식을 그대로 표시합니다.')
        else:
            parsed = parameter_expression(p['expression'], p['parameters'], ('s',))
            transformed = parsed.xreplace({symbols_for(('s',))[0]: s})
            try:
                time_expression = sp.inverse_laplace_transform(transformed, s, t)
            except Exception as exc:
                time_expression = sp.InverseLaplaceTransform(transformed, s, t, None)
                notices.append(f'역변환을 완료하지 못했습니다({type(exc).__name__}). 미평가 식을 표시합니다.')
            rows = [['F(s)', str(transformed)], ['f(t)', str(time_expression)],
                    ['해석', '인과적 단측 역변환입니다. 일반적인 양측 변환의 모든 ROC를 열거하지 않습니다.']]
        data = dict(time_expression=time_expression, transformed=transformed,
                    symbolic='\n\n'.join(f'{label}\n{value}' for label, value in rows))
        # 분포와 미평가 변환을 일반 함수의 0 값으로 그리지 않습니다.
        if time_expression.has(sp.DiracDelta, sp.InverseLaplaceTransform, sp.LaplaceTransform):
            notices.append('분포 또는 미평가 변환이 포함되어 일반 함수 그래프를 생략합니다. 기호 결과를 확인하십시오.')
        else:
            time = np.linspace(0, p['end_time'], p['num_points'])[1:]
            try:
                values = evaluate(time_expression, (time,), ('t',))
                data.update(time=time, values=values)
            except Exception as exc:
                notices.append(f'기호 결과는 계산되었지만 실수 그래프를 표시할 수 없습니다: {type(exc).__name__}')
        tables = {'기호 변환': (['항목', '결과'], rows)}
        if 'values' in data:
            tables['시간 표본'] = (['t', 'f(t)'], np.column_stack((data['time'], data['values'])))
        return Result(self.id, p, {'표시 표본 수': len(data.get('time', []))}, tables, data, notices)

    def figures(self, result):
        from matplotlib.figure import Figure
        figure = Figure(figsize=(8, 5), layout='constrained')
        axis = figure.subplots()
        if 'values' in result.data:
            axis.plot(result.data['time'], result.data['values'])
            axis.set(xlabel='t > 0', ylabel='f(t)', title='Time-domain function')
            axis.grid(alpha=.3)
        else:
            axis.text(.5, .5, 'Symbolic result only\nSee the symbolic transform table', ha='center', va='center')
            axis.set_axis_off()
        return {'시간 영역': figure}

"""고정 온도 경계조건을 갖는 1차원 열방정식의 변수분리 해입니다."""
import numpy as np
from engineering_math.core.expr import parameter_expression, evaluate
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.pde import sine_coefficients, heat_accuracy, evolution_figures


class HeatTopic:
    id = 'heat'
    title = '편미분방정식 · 1차원 열전도'
    description = 'u_t = α u_xx를 사인 급수로 계산합니다. 양 끝 온도는 시간에 관계없이 고정됩니다.'
    inputs = (
        InputSpec('initial', '초기 온도 u(x,0)', 'text', 'sin(pi*x/ell)', help='x와 막대 길이 ell을 사용하십시오.'),
        InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
        InputSpec('length', '막대 길이 ell', 'float', 1., .001, 1000),
        InputSpec('alpha', '열확산계수 α', 'float', .1, .000001, 1000),
        InputSpec('left', '왼쪽 경계 온도', 'float', 0., -100000, 100000),
        InputSpec('right', '오른쪽 경계 온도', 'float', 0., -100000, 100000),
        InputSpec('N', '사인 급수 차수 N', 'int', 20, 1, 100),
        InputSpec('num_points', '공간 표본 수', 'int', 501, 201, 5001),
        InputSpec('time_points', '시간 표본 수', 'int', 101, 2, 501),
        InputSpec('end_time', '마지막 시간', 'float', 1., .000001, 10000),
    )
    examples = {'기본 사인 모드': {}, '두 사인 모드': {'initial': 'sin(pi*x/ell)+0.5*sin(3*pi*x/ell)'},
                '삼각형 초기 온도': {'initial': '1-abs(2*x/ell-1)'},
                '정상 상태': {'initial': '20+60*x/ell', 'left': 20., 'right': 80.}}

    def compute(self, params):
        params = validate_params(self.inputs, params)
        length, count = params['length'], params['N']
        x = np.linspace(0, length, params['num_points'])
        time = np.linspace(0, params['end_time'], params['time_points'])
        expression = parameter_expression(params['initial'], params['parameters'], variables=('x', 'ell'))
        initial = evaluate(expression, (x, length), variables=('x', 'ell'))
        steady = params['left'] + (params['right']-params['left'])*x/length
        residual = initial-steady
        # 기확장의 사인 계수는 (2/ell) ∫[0,ell] residual*sin(nπx/ell) dx와 같습니다.
        coefficients = sine_coefficients(x, residual, length, count)
        wave_numbers = np.arange(1, count+1)*np.pi/length
        basis = np.sin(wave_numbers[:, None]*x)
        decay = np.exp(-params['alpha']*time[:, None]*wave_numbers**2)
        temperature = steady + decay @ (coefficients[:, None]*basis)
        temperature[:, 0], temperature[:, -1] = params['left'], params['right']
        if not np.all(np.isfinite(temperature)):
            raise ValueError('온도 계산에서 유한하지 않은 값이 발생했습니다.')
        energy = np.trapezoid((temperature-steady)**2, x=x, axis=1)
        accuracy = heat_accuracy(expression, params, coefficients)
        notices = ['길이·시간·열확산계수의 단위는 일관되게 입력하십시오. α의 차원은 길이²/시간입니다.',
                   '유한 차수의 변수분리 해입니다. 공간 표본은 계수 적분과 시각화에 사용됩니다.']
        notices.append('정확도 비교는 고정된 2049개 평가점에서 수행합니다. 격자·차수 간 차이는 참해에 대한 오차 상한이 아닙니다.')
        if not np.allclose(initial[[0, -1]], [params['left'], params['right']]):
            notices.append('초기 함수의 끝점이 경계조건과 다릅니다. 급수 해의 끝점은 지정한 경계 온도로 고정됩니다.')
        metrics = {'초기 재구성 RMSE': float(np.sqrt(np.mean((temperature[0]-initial)**2))),
                   '최종 최저 온도': float(temperature[-1].min()), '최종 최고 온도': float(temperature[-1].max()),
                   '격자 세분화 계수 변화': accuracy['coefficient_change']}
        tables = {'사인 계수': (['n', 'b_n'], np.column_stack((np.arange(1, count+1), coefficients))),
                  '잔차 에너지': (['t', 'integral (u-steady)^2 dx'], np.column_stack((time, energy))),
                  '온도': (['t', 'x', 'u'], np.column_stack((np.repeat(time, len(x)), np.tile(x, len(time)), temperature.ravel())))}
        tables['정확도 비교'] = (['변경 항목', '적분 표본 수', 'N', '초기 재구성 RMSE (공통 격자)', '기준과 차이 t=0 (최대)', '기준과 차이 t=마지막 (최대)'], accuracy['rows'])
        return Result(self.id, params, metrics, tables,
                      dict(x=x, time=time, initial=initial, steady=steady, temperature=temperature, coefficients=coefficients, accuracy=accuracy), notices)

    def figures(self, result):
        from matplotlib.figure import Figure
        data = result.data
        figure, heatmap = evolution_figures(data['x'], data['time'], data['initial'], data['temperature'],
                                           'Temperature', 'Heat equation')
        accuracy_figure = Figure(figsize=(8, 5), layout='constrained')
        axes = accuracy_figure.subplots(2, 1, sharex=True)
        accuracy = data['accuracy']
        axes[0].plot(accuracy['x'], accuracy['initial'], 'k--', label='Initial input')
        axes[0].plot(accuracy['x'], accuracy['reconstruction'], label='Sine reconstruction at t=0')
        axes[0].set(ylabel='Temperature', title='Initial reconstruction (not time evolution)')
        axes[0].legend()
        axes[1].plot(accuracy['x'], accuracy['initial']-accuracy['reconstruction'])
        axes[1].set(xlabel='x', ylabel='Initial error')
        for axis in axes:
            axis.grid(alpha=.3)
        return {'시간별 온도': figure, '시공간 히트맵': heatmap, '초기 재구성': accuracy_figure}

    def animation(self, result):
        data = result.data
        return dict(x=data['x'], frames=data['temperature'], labels=[f't = {t:.5g}' for t in data['time']],
                    reference=data['initial'], xlabel='x', ylabel='Temperature')

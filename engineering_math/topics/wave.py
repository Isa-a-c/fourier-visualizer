"""고정된 양 끝을 갖는 현의 무감쇠 파동방정식을 계산합니다."""
import numpy as np
from engineering_math.core.tables import GridRows
from engineering_math.core.expr import parameter_expression, evaluate
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.pde import sine_coefficients, sine_basis, evolution_figures, evolution_animation


class WaveTopic:
    id = 'wave'
    title = '12.2–12.3 · 현의 모델링과 변수분리'
    description = 'u_tt = c² u_xx의 고정단 현을 계산합니다. 초기 변위와 초기 속도를 각각 입력하십시오.'
    inputs = (
        InputSpec('initial', '초기 변위 u(x,0)', 'text', 'sin(pi*x/ell)'),
        InputSpec('velocity', '초기 속도 u_t(x,0)', 'text', '0'),
        InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
        InputSpec('length', '현의 길이 ell', 'float', 1., .001, 1000),
        InputSpec('speed', '파동 속력 c', 'float', 1., .000001, 1000),
        InputSpec('N', '사인 급수 차수 N', 'int', 20, 1, 100),
        InputSpec('num_points', '공간 표본 수', 'int', 501, 201, 5001),
        InputSpec('time_points', '시간 표본 수', 'int', 201, 2, 501),
        InputSpec('end_time', '마지막 시간', 'float', 2., .000001, 10000),
        InputSpec('comparison_alpha', '비교 열확산계수 α', 'float', .1, .000001, 1000,
                  help='열·파동 비교 그림에만 사용하며 파동 해에는 영향을 주지 않습니다.'),
    )
    examples = {
        '기본 사인 모드': {},
        '두 사인 모드': {'initial': 'sin(pi*x/ell)+0.5*sin(3*pi*x/ell)'},
        '삼각형 초기 변위': {'initial': '1-abs(2*x/ell-1)'},
        '초기 속도만 부여': {'initial': '0', 'velocity': 'sin(pi*x/ell)'},
    }

    def compute(self, params):
        params = validate_params(self.inputs, params)
        length, count = params['length'], params['N']
        x = np.linspace(0, length, params['num_points'])
        time = np.linspace(0, params['end_time'], params['time_points'])
        displacement_expr = parameter_expression(params['initial'], params['parameters'], ('x', 'ell'))
        velocity_expr = parameter_expression(params['velocity'], params['parameters'], ('x', 'ell'))
        initial = evaluate(displacement_expr, (x, length), ('x', 'ell'))
        initial_velocity = evaluate(velocity_expr, (x, length), ('x', 'ell'))
        displacement_coefficients = sine_coefficients(x, initial, length, count)
        velocity_coefficients = sine_coefficients(x, initial_velocity, length, count)
        wave_numbers = np.arange(1, count+1)*np.pi/length
        omega = params['speed']*wave_numbers
        phase = time[:, None]*omega
        # u_n(t)=A_n cos(ω_n t)+(V_n/ω_n)sin(ω_n t)입니다.
        amplitudes = (np.cos(phase)*displacement_coefficients
                      + np.sin(phase)*(velocity_coefficients/omega))
        derivatives = (-np.sin(phase)*(omega*displacement_coefficients)
                       + np.cos(phase)*velocity_coefficients)
        basis = sine_basis(x, length, count)
        displacement = amplitudes @ basis
        displacement[:, [0, -1]] = 0.
        reconstructed_velocity = velocity_coefficients @ basis
        reconstructed_velocity[[0, -1]] = 0.
        # 직교성을 이용한 단위 선밀도 에너지: 1/2 ∫(u_t²+c²u_x²)dx입니다.
        energy = length/4*np.sum(derivatives**2+(amplitudes*omega)**2, axis=1)
        heat = np.exp(-params['comparison_alpha']*time[:, None]*wave_numbers**2) @ (displacement_coefficients[:, None]*basis)
        heat[:, [0, -1]] = 0.
        if not all(np.all(np.isfinite(values)) for values in (displacement, energy, heat)):
            raise ValueError('유한하지 않은 결과가 발생했습니다. 입력 크기를 줄이십시오.')
        notices = [
            '무감쇠·무외력·일정한 파동 속력과 고정단 u(0,t)=u(ell,t)=0을 가정합니다.',
            '열 비교는 동일한 초기 모양과 0 경계조건을 사용합니다. 초기 속도는 파동에만 적용됩니다.',
            '시간 간격은 해의 적분 간격이 아니라 출력 간격입니다. 너무 크면 진동을 놓칠 수 있습니다.',
        ]
        if not np.allclose(initial[[0, -1]], 0) or not np.allclose(initial_velocity[[0, -1]], 0):
            notices.append('초기 변위 또는 속도의 끝점이 고정단 조건과 다릅니다. 급수 해의 끝점은 0으로 고정됩니다.')
        samples_per_period = 2*np.pi/(omega[-1]*(time[1]-time[0]))
        if samples_per_period < 10:
            notices.append(f'최고 차수 모드의 한 주기당 출력 표본이 {samples_per_period:.3g}개입니다. 시간 표본 수를 늘리거나 마지막 시간을 줄이십시오.')
        drift = float(np.max(np.abs(energy-energy[0])))
        metrics = {'초기 변위 RMSE': float(np.sqrt(np.mean((displacement[0]-initial)**2))),
                   '초기 속도 RMSE': float(np.sqrt(np.mean((reconstructed_velocity-initial_velocity)**2))),
                   '에너지 최대 변화': drift}
        tables = {
            '모드 계수': (['n', 'A_n', 'V_n', 'omega_n'],
                        np.column_stack((np.arange(1, count+1), displacement_coefficients, velocity_coefficients, omega))),
            '에너지': (['t', 'energy'], np.column_stack((time, energy))),
            '변위 및 열 비교': (['t', 'x', 'wave u', 'heat u'],
                              GridRows((time, x), (displacement, heat))),
        }
        return Result(self.id, params, metrics, tables,
                      dict(x=x, time=time, initial=initial, displacement=displacement, heat=heat,
                           energy=energy, velocity_initial=reconstructed_velocity,
                           displacement_coefficients=displacement_coefficients, velocity_coefficients=velocity_coefficients), notices)

    def figures(self, result):
        from matplotlib.figure import Figure
        data = result.data
        curve, heatmap = evolution_figures(data['x'], data['time'], data['initial'], data['displacement'],
                                          'Displacement', 'Fixed-end wave equation')
        comparison = Figure(figsize=(8, 6), layout='constrained')
        axes = comparison.subplots(2, 1, sharex=True)
        limit = max(float(np.max(np.abs(data['heat']))), float(np.max(np.abs(data['displacement']))), 1e-9)
        for axis, values, title in zip(axes, (data['heat'], data['displacement']), ('Heat: decay', 'Wave: oscillation')):
            for index in np.unique(np.linspace(0, len(data['time'])-1, 5, dtype=int)):
                axis.plot(data['x'], values[index], label=f"t={data['time'][index]:.3g}")
            axis.set(title=title, ylabel='u', ylim=(-1.05*limit, 1.05*limit))
            axis.grid(alpha=.3)
            axis.legend(fontsize=8, ncols=3)
        axes[-1].set_xlabel('x')
        energy_figure = Figure(figsize=(8, 4), layout='constrained')
        axis = energy_figure.subplots()
        axis.plot(data['time'], data['energy'])
        axis.set(xlabel='t', ylabel='Energy', title='Modal energy (unit linear density)')
        axis.grid(alpha=.3)
        return {'시간별 변위': curve, '시공간 히트맵': heatmap, '열·파동 비교': comparison, '에너지': energy_figure}

    def animation(self, result):
        data = result.data
        return evolution_animation(data['x'], data['time'], data['initial'], data['displacement'], 'Displacement')

"""선형 질량–스프링–댐퍼: 수치해와 행렬 지수의 독립 비교입니다."""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from engineering_math.core.models import InputSpec, Result, validate_params

FORCES = ('외력 없음', '일정한 외력', '사인 외력', '지연 계단 외력')


class OdeTopic:
    id = 'ode'
    title = '상미분방정식 · 질량–스프링–댐퍼'
    description = 'm y″ + c y′ + k y = F(t)의 초기값 문제와 위상평면을 살펴보십시오.'
    supports_animation = False
    inputs = (
        InputSpec('mass', '질량 m', 'float', 1., .01, 100),
        InputSpec('damping', '감쇠계수 c', 'float', .4, 0, 100),
        InputSpec('stiffness', '강성 k', 'float', 4., .01, 100),
        InputSpec('position', '초기 변위 y(0)', 'float', 1., -100, 100),
        InputSpec('velocity', '초기 속도 y′(0)', 'float', 0., -100, 100),
        InputSpec('mode', '외력 종류', 'choice', FORCES[0], choices=FORCES),
        InputSpec('amplitude', '외력 크기', 'float', 1., -100, 100, visible_in=FORCES[1:]),
        InputSpec('frequency', '외력 각주파수 (rad/s)', 'float', 2., .001, 100, visible_in=(FORCES[2],)),
        InputSpec('delay', '외력 시작 시간', 'float', 1., 0, 100, visible_in=(FORCES[3],)),
        InputSpec('end_time', '마지막 시간', 'float', 10., .01, 100),
        InputSpec('num_points', '출력 표본 수', 'int', 1001, 101, 5001),
    )
    examples = {'부족감쇠': {}, '무감쇠': {'damping': 0.},
                '임계감쇠': {'damping': 4.}, '과감쇠': {'damping': 6.},
                '공진 (무감쇠)': {'damping': 0., 'position': 0., 'mode': FORCES[2]},
                '지연 계단 응답': {'position': 0., 'mode': FORCES[3]}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        m, c, k = p['mass'], p['damping'], p['stiffness']
        time = np.linspace(0, p['end_time'], p['num_points'])
        natural = np.sqrt(k/m)
        ratio = c/(2*np.sqrt(m*k))
        kind = '무감쇠' if c == 0 else '임계감쇠' if np.isclose(ratio, 1., rtol=1e-8, atol=1e-10) else '부족감쇠' if ratio < 1 else '과감쇠'

        def force(t):
            if p['mode'] == FORCES[0]:
                return np.zeros_like(t, dtype=float)
            if p['mode'] == FORCES[1]:
                return np.full_like(t, p['amplitude'], dtype=float)
            if p['mode'] == FORCES[2]:
                return p['amplitude']*np.sin(p['frequency']*t)
            return p['amplitude']*(np.asarray(t) >= p['delay'])

        # 출력 간격과 내부 적분 간격은 구분합니다. 계단은 정확한 시작점에서 분할합니다.
        boundaries = [0., p['end_time']]
        if p['mode'] == FORCES[3] and 0 < p['delay'] < p['end_time']:
            boundaries.insert(1, p['delay'])
        step = min(p['end_time']/100, .2/max(natural, c/m, p['frequency'] if p['mode'] == FORCES[2] else 0))
        if p['end_time']/step > 200000:
            raise ValueError('계산 규모가 큽니다. 시간 범위 또는 계수 비율을 줄이십시오.')
        values = np.empty((2, len(time)))
        initial = np.array([p['position'], p['velocity']])
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            constant_force = float(force((start+stop)/2)) if p['mode'] != FORCES[2] else None
            def derivative(t, state):
                load = constant_force if constant_force is not None else float(force(t))
                return [state[1], (load-c*state[1]-k*state[0])/m]
            solution = solve_ivp(derivative, (start, stop), initial, rtol=1e-9, atol=1e-11,
                                 max_step=step, dense_output=True)
            if not solution.success:
                raise ValueError('수치 적분에 실패했습니다: '+solution.message)
            mask = (time >= start) & (time <= stop)
            values[:, mask] = solution.sol(time[mask])
            initial = solution.y[:, -1]

        # 외력을 추가 상태로 표현하여 행렬 지수의 정확한 선형 전파와 비교합니다.
        matrix = np.zeros((4, 4))
        matrix[:2, :2] = [[0, 1], [-k/m, -c/m]]
        matrix[1, 2] = 1/m
        if p['mode'] == FORCES[2]:
            matrix[2, 3] = p['frequency']
            matrix[3, 2] = -p['frequency']
        state = np.array([p['position'], p['velocity'], float(force(0.)),
                          p['amplitude'] if p['mode'] == FORCES[2] else 0.])
        reference = np.empty_like(values)
        reference[:, 0] = state[:2]
        propagation = expm(matrix*(time[1]-time[0]))
        for index in range(1, len(time)):
            before, after = time[index-1], time[index]
            if p['mode'] == FORCES[3] and before < p['delay'] <= after:
                state = expm(matrix*(p['delay']-before)) @ state
                state[2] = p['amplitude']
                state = expm(matrix*(after-p['delay'])) @ state
            else:
                state = propagation @ state
            reference[:, index] = state[:2]
        energy = .5*m*values[1]**2+.5*k*values[0]**2
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(reference)):
            raise ValueError('유한하지 않은 결과가 발생했습니다. 입력 범위를 줄이십시오.')

        # 초기조건을 포함한 단측 라플라스 변환 식입니다.
        s = sp.Symbol('s')
        mass, damping, stiffness, y0, v0, amplitude, frequency, delay = map(
            lambda value: sp.Rational(str(value)), (m, c, k, p['position'], p['velocity'], p['amplitude'], p['frequency'], p['delay']))
        forcing = [sp.S.Zero, amplitude/s, amplitude*frequency/(s**2+frequency**2), amplitude*sp.exp(-delay*s)/s][FORCES.index(p['mode'])]
        transformed = (forcing+mass*(s*y0+v0)+damping*y0)/(mass*s**2+damping*s+stiffness)
        tables = {'시간 응답': (['t', 'y', 'velocity', 'reference y', 'force', 'energy'],
                              np.column_stack((time, values.T, reference[0], force(time), energy))),
                  '라플라스 연결': (['항목', '식'], [['F(s)', str(forcing)], ['Y(s)', str(transformed)],
                               ['초기조건', 'm(s²Y−s y₀−v₀)+c(sY−y₀)+kY=F(s)'],
                               ['검증 기준', '독립 기준 해는 증강 상태행렬의 지수로 계산합니다.']])}
        return Result(self.id, p, {'고유 각주파수': natural, '감쇠비': ratio,
                                  '변위 최대 비교 오차': float(np.max(np.abs(values[0]-reference[0]))),
                                  '속도 최대 비교 오차': float(np.max(np.abs(values[1]-reference[1])))}, tables,
                      dict(time=time, values=values, reference=reference, energy=energy, force=force(time),
                           laplace_input=str(transformed), regime=kind,
                           symbolic='\n\n'.join(f'{label}\n{value}' for label, value in tables['라플라스 연결'][1])),
                      [f'감쇠 분류: {kind}. 단위는 서로 일관되게 입력하십시오.',
                       '출력 표본 수는 내부 적분 횟수와 다릅니다. 빠른 진동은 출력 표본이 적으면 잘 보이지 않을 수 있습니다.',
                       '계단 외력의 시작점에서 적분을 분할합니다. 라플라스 연결 버튼으로 Y(s)의 역변환을 확인하십시오.'])

    def figures(self, result):
        from matplotlib.figure import Figure
        d = result.data
        response = Figure(figsize=(8, 6), layout='constrained')
        axes = response.subplots(2, 1, sharex=True)
        for axis, index, label in zip(axes, (0, 1), ('Displacement', 'Velocity')):
            axis.plot(d['time'], d['values'][index], label='solve_ivp')
            axis.plot(d['time'], d['reference'][index], '--', label='Matrix exponential')
            axis.set(ylabel=label)
            axis.legend()
            axis.grid(alpha=.3)
        axes[-1].set_xlabel('t')
        phase = Figure(figsize=(8, 5), layout='constrained')
        axis = phase.subplots()
        axis.plot(*d['values'])
        axis.plot(*d['values'][:, 0], 'go', label='Initial')
        axis.plot(*d['values'][:, -1], 'rx', label='Final')
        axis.set(xlabel='Displacement', ylabel='Velocity', title='Phase plane')
        axis.grid(alpha=.3)
        axis.legend()
        energy = Figure(figsize=(8, 5), layout='constrained')
        axes = energy.subplots(2, 1, sharex=True)
        axes[0].plot(d['time'], d['energy'])
        axes[0].set_ylabel('Mechanical energy')
        axes[1].plot(d['time'], d['force'])
        axes[1].set(xlabel='t', ylabel='Force')
        for axis in axes:
            axis.grid(alpha=.3)
        return {'시간 응답': response, '위상평면': phase, '에너지·외력': energy}

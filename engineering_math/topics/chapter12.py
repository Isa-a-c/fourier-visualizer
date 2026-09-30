"""교재 12장의 입력 명세·결과·시각화입니다. 계산은 core에 분리합니다."""
import numpy as np
from engineering_math.core.tables import GridRows
import sympy as sp
from scipy.special import erfc
from matplotlib.figure import Figure
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.expr import parameter_expression, evaluate, symbols_for
from engineering_math.core.chapter11 import checked
from engineering_math.core.chapter12 import (dalembert, infinite_heat, rectangle_membrane,
    disk_membrane, disk_potential, sphere_potential)
from engineering_math.core.pde import evolution_figures, evolution_animation

PARAMETERS = InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', '')
TIME_INPUTS = (
    InputSpec('end_time', '마지막 시간', 'float', 2., .001, 20),
    InputSpec('time_points', '시간 표본 수', 'int', 81, 11, 201),
)


def function(text, parameters='', variables=('x',)):
    expression = parameter_expression(text, parameters, variables)
    return lambda *values: evaluate(expression, values, variables)


def evolution_result(topic_id, p, x, time, frames, initial, notices, metrics=None):
    checked(frames)
    statistics = {'최대 |u|': float(np.max(np.abs(frames))),
                  '초기 재구성 RMSE': float(np.sqrt(np.mean((frames[0] - initial) ** 2)))}
    statistics.update(metrics or {})
    checked(list(statistics.values()))
    tables = {'시공간 표본': (['t', 'x', 'u'], GridRows((time, x), (frames,)))}
    return Result(topic_id, p, statistics, tables, dict(x=x, time=time, frames=frames, initial=initial), notices)


class EvolutionTopic:
    def figures(self, result):
        d = result.data
        curve, heatmap = evolution_figures(d['x'], d['time'], d['initial'], d['frames'], 'u', self.id)
        return {'시간별 함수': curve, '시공간 히트맵': heatmap}

    def animation(self, result):
        d = result.data
        return evolution_animation(d['x'], d['time'], d['initial'], d['frames'])


# ── 12.1: 후보 해를 방정식에 대입하여 잔차 확인 ────────────────────
class PdeBasicsTopic:
    id = 'pde_basics'
    title = '12.1 · PDE 기본개념과 해 검산'
    description = '후보 u(x,y,t)를 기호 미분하고 방정식의 잔차를 확인합니다. 초기·경계조건은 별도로 확인해야 합니다.'
    supports_animation = False
    inputs = (
        InputSpec('mode', '방정식', 'choice', 'Laplace', choices=('Laplace', 'Poisson', '열', '파동')),
        InputSpec('candidate', '후보 해 u(x,y,t)', 'text', 'x^2-y^2'),
        InputSpec('forcing', 'Poisson 우변 f(x,y,t)', 'text', '0', visible_in=('Poisson',)), PARAMETERS,
        InputSpec('alpha', '열확산계수 α', 'float', .1, .000001, 100, visible_in=('열',)),
        InputSpec('speed', '파동 속력 c', 'float', 1., .001, 100, visible_in=('파동',)),
        InputSpec('time', '검산 시간 t', 'float', .5, 0, 20),
        InputSpec('extent', '표시 구간 [-X,X]', 'float', 1., .01, 10),
    )
    examples = {'조화함수': {}, 'Poisson 다항식': {'mode': 'Poisson', 'candidate': 'x^2+y^2', 'forcing': '4'},
                '열방정식 해': {'mode': '열', 'candidate': 'exp(-0.1*pi^2*t)*sin(pi*x)'},
                '진행파': {'mode': '파동', 'candidate': 'sin(x-t)'}, '해가 아닌 함수': {'candidate': 'x^2+y^2'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x, y, t = symbols_for(('x', 'y', 't'))
        u = parameter_expression(p['candidate'], p['parameters'], ('x', 'y', 't'))
        if p['mode'] in ('Laplace', 'Poisson'):
            residual = sp.diff(u, x, 2) + sp.diff(u, y, 2)
            if p['mode'] == 'Poisson':
                residual -= parameter_expression(p['forcing'], p['parameters'], ('x', 'y', 't'))
        elif p['mode'] == '열':
            residual = sp.diff(u, t) - sp.Rational(str(p['alpha'])) * sp.diff(u, x, 2)
        else:
            residual = sp.diff(u, t, 2) - sp.Rational(str(p['speed'])) ** 2 * sp.diff(u, x, 2)
        residual = sp.simplify(residual)
        grid = np.linspace(-p['extent'], p['extent'], 81)
        xx, yy = np.meshgrid(grid, grid, indexing='ij')
        values = evaluate(u, (xx, yy, p['time']), ('x', 'y', 't'))
        errors = evaluate(residual, (xx, yy, p['time']), ('x', 'y', 't'))
        text = f'후보 u = {u}\n\n방정식 잔차 = {residual}\n\n기호적으로 0: {residual == 0}'
        return Result(self.id, p, {'표본 잔차 최대값': float(np.max(np.abs(errors)))},
            {'기호 검산': (['항목', '값'], [['u', str(u)], ['잔차', str(residual)], ['기호적으로 0', str(residual == 0)]])},
            dict(x=grid, y=grid, field=values, residual=errors, symbolic=text),
            ['표본 잔차가 작다는 사실만으로 전체 영역의 해임을 증명할 수 없습니다.',
             '기호 잔차가 0이어도 특이점의 정의역과 초기·경계조건을 별도로 확인하십시오. 분포를 포함한 후보는 수치 검산을 지원하지 않습니다.'])

    def figures(self, result):
        d = result.data
        return {'후보와 잔차': field_figure(d['x'], d['y'], [d['field'], d['residual']], ['Candidate', 'PDE residual'])}


# ── 12.4, 12.7: 무한 공간에서의 해 ────────────────────────────────
class DalembertTopic(EvolutionTopic):
    id = 'dalembert'
    title = '12.4 · D’Alembert 해와 특성선'
    description = '무한 현의 초기 변위·속도를 입력합니다. 화면 끝은 경계가 아니므로 반사되지 않습니다.'
    inputs = (InputSpec('initial', '초기 변위 f(x)', 'text', 'exp(-x^2)'),
              InputSpec('velocity', '초기 속도 g(x)', 'text', '0'), PARAMETERS,
              InputSpec('speed', '파동 속력 c', 'float', 1., .001, 20),
              InputSpec('extent', '표시 반폭 X', 'float', 5., .1, 100),
              InputSpec('num_points', '공간 표본 수', 'int', 501, 101, 2001),
              InputSpec('quadrature_points', '속도 적분 표본 수', 'int', 8001, 1001, 32001)) + TIME_INPUTS
    examples = {'양방향 Gaussian': {}, '속도만 부여': {'initial': '0', 'velocity': 'exp(-x^2)'},
                '오른쪽 진행파': {'initial': 'sin(x)', 'velocity': '-cos(x)'},
                '삼각 펄스': {'initial': 'Piecewise((1-abs(x),abs(x)<1),(0,True))'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x = np.linspace(-p['extent'], p['extent'], p['num_points'])
        time = np.linspace(0, p['end_time'], p['time_points'])
        f, g = function(p['initial'], p['parameters']), function(p['velocity'], p['parameters'])
        frames, grid = dalembert(f, g, x, time, p['speed'], p['quadrature_points'])
        return evolution_result(self.id, p, x, time, frames, f(x),
            ['u=[f(x−ct)+f(x+ct)]/2 + (1/2c)∫[x−ct,x+ct]g(s)ds입니다.',
             f'초기 함수는 표시 구간 밖 [{grid[0]:g}, {grid[-1]:g}]에서도 정의되어야 합니다.',
             '속도 적분은 균일 격자의 누적 사다리꼴 적분과 보간입니다. 좁은 펄스·불연속에서는 적분 표본 수를 늘리십시오.'])

    def figures(self, result):
        figures = super().figures(result)
        d, p = result.data, result.params
        figure = Figure(figsize=(8, 5), layout='constrained')
        axis = figure.subplots()
        axis.plot(p['speed'] * d['time'], d['time'], label='x=ct')
        axis.plot(-p['speed'] * d['time'], d['time'], label='x=-ct')
        axis.set(xlabel='x', ylabel='t', title='Characteristics through the origin')
        axis.legend(); axis.grid(alpha=.3)
        figures['특성선'] = figure
        return figures


class InfiniteHeatTopic(EvolutionTopic):
    id = 'infinite_heat'
    title = '12.7 · 무한 막대와 열핵'
    description = '무한 막대의 열핵 합성곱을 계산합니다. Fourier 변환 해 F(ω,t)=F(ω,0)exp(−αω²t)와 동등합니다.'
    inputs = (InputSpec('initial', '초기 온도 f(x)', 'text', 'exp(-x^2)'), PARAMETERS,
              InputSpec('alpha', '열확산계수 α', 'float', .2, .000001, 10),
              InputSpec('extent', '표시 반폭 X', 'float', 5., .1, 100),
              InputSpec('num_points', '공간 표본 수', 'int', 401, 101, 1001),
              InputSpec('order', 'Gauss–Hermite 구적 차수', 'int', 80, 16, 160)) + TIME_INPUTS
    examples = {'Gaussian 확산': {}, '상수 보존': {'initial': '3'},
                '사각 펄스': {'initial': 'Piecewise((1,abs(x)<1),(0,True))'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x = np.linspace(-p['extent'], p['extent'], p['num_points'])
        time = np.linspace(0, p['end_time'], p['time_points'])
        f = function(p['initial'], p['parameters'])
        frames = infinite_heat(f, x, time, p['alpha'], p['order'])
        refined = infinite_heat(f, x, time[-1:], p['alpha'], 2 * p['order'])[0]
        result = evolution_result(self.id, p, x, time, frames, f(x),
            ['열핵 u=(1/√(4παt))∫ exp(−(x−ξ)²/(4αt))f(ξ)dξ를 변수변환하여 구적합니다.',
             '표시 경계에서 온도를 고정하지 않습니다. 함수는 실수축 전체에 정의되어야 합니다.',
             '구적 차수 두 배와의 차이는 오차 상한이 아닙니다. 불연속·좁은 펄스는 수렴이 느릴 수 있습니다.'],
            {'구적 세분화 최종 최대 차이': float(np.max(np.abs(frames[-1] - refined)))})
        mass = np.trapezoid(frames, x=x, axis=1)
        checked(mass)
        result.tables['표시 구간 내 적분'] = (['t', 'integral over displayed interval'], np.column_stack((time, mass)))
        return result


# ── 12.8~12.11의 공통 2차원 그림 ──────────────────────────────────
def field_figure(x, y, fields, titles, polar=False):
    figure = Figure(figsize=(9, 5), layout='constrained')
    axes = figure.subplots(1, len(fields), squeeze=False)[0]
    for axis, values, title in zip(axes, fields, titles):
        if polar:
            rr, tt = np.meshgrid(y, x, indexing='ij')
            # θ 주기 격자에 끝점을 추가하여 원판의 이음매를 닫습니다.
            xx, yy = rr * np.cos(tt), rr * np.sin(tt)
            mesh = axis.contourf(np.c_[xx, xx[:, :1]], np.c_[yy, yy[:, :1]],
                                 np.c_[values, values[:, :1]], levels=24, cmap='coolwarm')
        else:
            mesh = axis.pcolormesh(x, y, values.T, shading='auto', cmap='coolwarm')
        axis.set(title=title, xlabel='x', ylabel='y', aspect='equal')
        figure.colorbar(mesh, ax=axis, label='u')
    return figure


class MembraneTopic:
    id = 'rectangle_membrane'
    title = '12.8–12.9 · 직사각형 막과 이중 급수'
    description = '가장자리가 고정된 균질 막의 u_tt=c²(u_xx+u_yy)를 이중 사인 급수로 계산합니다.'
    inputs = (InputSpec('initial', '초기 변위 f(x,y)', 'text', 'sin(pi*x/ell)*sin(pi*y/height)'),
              InputSpec('velocity', '초기 속도 g(x,y)', 'text', '0'), PARAMETERS,
              InputSpec('length', '가로 ell', 'float', 1., .05, 20),
              InputSpec('height', '세로 height', 'float', 1., .05, 20),
              InputSpec('speed', '파동 속력 c', 'float', 1., .01, 20),
              InputSpec('N', '방향별 모드 수 N', 'int', 6, 1, 16),
              InputSpec('num_points', '방향별 공간 표본 수', 'int', 81, 41, 161)) + TIME_INPUTS
    examples = {'기본 모드': {}, '두 모드': {'initial': 'sin(pi*x/ell)*sin(pi*y/height)+0.4*sin(2*pi*x/ell)*sin(3*pi*y/height)'},
                '속도만 부여': {'initial': '0', 'velocity': 'sin(pi*x/ell)*sin(pi*y/height)'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x = np.linspace(0, p['length'], p['num_points'])
        y = np.linspace(0, p['height'], p['num_points'])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        variables = ('x', 'y', 'ell', 'height')
        initial = function(p['initial'], p['parameters'], variables)(xx, yy, p['length'], p['height'])
        velocity = function(p['velocity'], p['parameters'], variables)(xx, yy, p['length'], p['height'])
        time = np.linspace(0, p['end_time'], p['time_points'])
        frames, energy, a, v, omega = rectangle_membrane(x, y, initial, velocity,
                    p['length'], p['height'], p['speed'], p['N'], time)
        modes = [[m + 1, n + 1, a[m, n], v[m, n], omega[m, n]] for m in range(p['N']) for n in range(p['N'])]
        result = self.make_result(p, x, y, time, frames, initial, energy,
                    (['m', 'n', 'A_mn', 'V_mn', 'omega'], modes))
        if max(np.max(np.abs(initial[[0,-1], :])), np.max(np.abs(initial[:, [0,-1]])),
               np.max(np.abs(velocity[[0,-1], :])), np.max(np.abs(velocity[:, [0,-1]]))) > 1e-8:
            result.notices.append('초기 변위·속도의 가장자리가 고정단 조건과 다릅니다. 급수 해의 가장자리는 0입니다.')
        result.data['omega'] = omega
        self.warn_time_sampling(result, omega)
        return result

    @staticmethod
    def warn_time_sampling(result, omega):
        time = result.data['time']
        samples = 2*np.pi/(float(np.max(omega))*(time[1]-time[0]))
        if samples < 10:
            result.notices.append(f'최고 모드의 주기당 출력 표본이 {samples:.3g}개입니다. 시간 표본 수를 늘리거나 마지막 시간을 줄이십시오.')

    def make_result(self, p, x, y, time, frames, initial, energy, modes, polar=False):
        metrics = {'초기 재구성 RMSE': float(np.sqrt(np.mean((frames[0]-initial)**2))),
                   '에너지 최대 변화': float(np.max(np.abs(energy-energy[0])))}
        checked(list(metrics.values()))
        data = dict(x=x, y=y, time=time, frames=frames, initial=initial, energy=energy, polar=polar)
        samples = GridRows((y, x), (frames[-1],), (1, 0)) if polar else GridRows((x, y), (frames[-1],))
        tables = {'모드 계수': modes, '에너지': (['t', 'energy'], np.column_stack((time, energy))),
                  '최종 공간 표본': (['theta' if polar else 'x', 'r' if polar else 'y', 'u'], samples)}
        return Result(self.id, p, metrics, tables, data,
            ['고정 가장자리·일정한 속력·무감쇠·무외력의 유한 모드 해입니다. 임의 형태의 막은 지원하지 않습니다.',
             '에너지는 단위 면밀도 모드 에너지입니다. 출력 시간 간격이 크면 진동을 놓칠 수 있습니다.'])

    def figures(self, result):
        d = result.data
        figures = {'막의 변위': field_figure(d['x'], d['y'], [d['frames'][0], d['frames'][-1]],
                            ['t=0 (projection)', f"t={d['time'][-1]:g}"], d['polar'])}
        figure = Figure(figsize=(8, 4), layout='constrained')
        axis = figure.subplots()
        axis.plot(d['time'], d['energy'])
        axis.set(xlabel='t', ylabel='Energy', title='Modal energy'); axis.grid(alpha=.3)
        figures['에너지'] = figure
        return figures

    def animation(self, result):
        d = result.data
        # 극좌표 재생에서도 마지막 각도와 첫 각도를 연결하여 이음매를 닫습니다.
        x = np.r_[d['x'], 2*np.pi] if d['polar'] else d['x']
        frames = np.concatenate((d['frames'], d['frames'][:, :, :1]), axis=2) if d['polar'] else d['frames'].transpose(0,2,1)
        return dict(kind='field', x=x, y=d['y'], polar=d['polar'], frames=frames,
                    labels=[f't={t:.5g}' for t in d['time']], xlabel='x', ylabel='y')


class DiskMembraneTopic(MembraneTopic):
    id = 'disk_membrane'
    title = '12.10 · 원형 막과 Fourier–Bessel 급수'
    description = '고정 원주를 갖는 원형 막입니다. r·theta·ell을 입력에 사용하며 ell은 반지름입니다.'
    inputs = (InputSpec('initial', '초기 변위 f(r,theta)', 'text', '1-(r/ell)^2'),
              InputSpec('velocity', '초기 속도 g(r,theta)', 'text', '0'), PARAMETERS,
              InputSpec('radius', '반지름 ell', 'float', 1., .05, 20),
              InputSpec('speed', '파동 속력 c', 'float', 1., .01, 20),
              InputSpec('radial_count', '방사 모드 수', 'int', 6, 1, 16),
              InputSpec('angular_count', '최대 각도 차수 m', 'int', 2, 0, 6),
              InputSpec('num_points', '방사 표본 수', 'int', 101, 41, 201),
              InputSpec('angular_points', '각도 표본 수', 'int', 81, 33, 161)) + TIME_INPUTS
    examples = {'축대칭 포물선': {}, '비축대칭 막': {'initial': '(r/ell)*(1-r/ell)*cos(theta)'},
                '기본 Bessel 모드': {'initial': 'besselj(0,2.404825557695773*r/ell)', 'radial_count': 1, 'angular_count': 0}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        r = np.linspace(0, p['radius'], p['num_points'])
        theta = np.linspace(0, 2*np.pi, p['angular_points'], endpoint=False)
        rr, tt = np.meshgrid(r, theta, indexing='ij')
        variables = ('r', 'theta', 'ell')
        initial = function(p['initial'], p['parameters'], variables)(rr, tt, p['radius'])
        velocity = function(p['velocity'], p['parameters'], variables)(rr, tt, p['radius'])
        if not np.allclose(initial[0], initial[0,0]) or not np.allclose(velocity[0], velocity[0,0]):
            raise ValueError('r=0에서 함수값은 theta에 관계없이 하나여야 합니다.')
        time = np.linspace(0, p['end_time'], p['time_points'])
        frames, energy, a, v, omega, labels = disk_membrane(r, theta, initial, velocity,
            p['radius'], p['speed'], p['radial_count'], p['angular_count'], time)
        rows = [[*label, a[i], v[i], omega[i]] for i, label in enumerate(labels)]
        result = self.make_result(p, theta, r, time, frames, initial, energy,
            (['m', 'n', 'angular', 'A', 'V', 'omega'], rows), polar=True)
        if not np.allclose(initial[-1], 0) or not np.allclose(velocity[-1], 0):
            result.notices.append('입력 원주 값이 고정 경계 0과 다릅니다. 모드 해의 원주는 0입니다.')
        result.data['omega'] = omega
        self.warn_time_sampling(result, omega)
        return result


# ── 12.11: 원통·구 좌표에서의 내부 퍼텐셜 ──────────────────────────
class PotentialTopic:
    id = 'coordinate_potential'
    title = '12.11 · 원통·구 좌표의 Laplace 퍼텐셜'
    description = '원판 내부(z 독립 원통) 또는 축대칭 구 내부의 Dirichlet 경계값 문제를 계산합니다.'
    supports_animation = False
    inputs = (InputSpec('mode', '영역', 'choice', '원판 · 원통 z 독립', choices=('원판 · 원통 z 독립', '구 · 축대칭')),
              InputSpec('boundary', '경계값 h(theta) 또는 h(mu)', 'text', 'cos(2*theta)'), PARAMETERS,
              InputSpec('radius', '반지름 R', 'float', 1., .05, 20),
              InputSpec('N', '최대 차수 N', 'int', 8, 1, 30),
              InputSpec('num_points', '경계 적분 표본 수', 'int', 1001, 201, 8001))
    examples = {'원판 사중극': {}, '원판 상수': {'boundary': '3'},
                '구 쌍극': {'mode': '구 · 축대칭', 'boundary': 'mu'},
                '구 사중극': {'mode': '구 · 축대칭', 'boundary': '(3*mu^2-1)/2'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        r = np.linspace(0, p['radius'], 81)
        if p['mode'].startswith('원판'):
            theta = np.linspace(0, 2*np.pi, p['num_points'], endpoint=False)
            boundary = function(p['boundary'], p['parameters'], ('theta',))
            field, constant, coefficients = disk_potential(boundary, r, theta, p['radius'], p['N'])
            target = boundary(theta)
            rows = [[0, constant, 0], *coefficients]
            columns = ['n', 'a_n', 'b_n']
        else:
            theta = np.linspace(0, 2*np.pi, 241, endpoint=False)
            boundary = function(p['boundary'], p['parameters'], ('mu',))
            field, coefficients = sphere_potential(boundary, r, theta, p['radius'], p['N']+1, p['num_points'])
            target = boundary(np.cos(theta))
            rows, columns = list(enumerate(coefficients)), ['n', 'c_n']
        # 실제 그림은 최대 241개 각도 표본만 사용하여 UI 부담을 줄입니다.
        stride = max(1, int(np.ceil(len(theta)/241)))
        rmse = float(np.sqrt(np.mean(checked((field[-1]-target)**2))))
        return Result(self.id, p, {'경계 재구성 RMSE': rmse},
            {'계수': (columns, rows), '퍼텐셜': (['r', 'theta', 'u'], GridRows((r, theta), (field,)))},
            dict(x=theta[::stride], y=r, field=field[:, ::stride], full_field=field, theta=theta, r=r),
            ['원판: u=a₀+Σ(r/R)ⁿ(aₙcos nθ+bₙsin nθ). 구: u=Σcₙ(r/R)ⁿPₙ(cos φ).',
             '중심에서 유한한 내부해와 Dirichlet 조건만 지원합니다. 구 그림은 대칭축을 포함하는 단면입니다.',
             '일반 3차원 원통·비축대칭 구·외부 영역·Neumann·Robin 조건은 지원하지 않습니다.'])

    def figures(self, result):
        d = result.data
        figure = field_figure(d['x'], d['y'], [d['field']], ['Interior potential (section)'], True)
        if result.params['mode'].startswith('구'):
            figure.axes[0].set(xlabel='z (symmetry axis)', ylabel='Signed radial coordinate')
        return {'퍼텐셜 단면': figure}


# ── 12.12: Laplace 변환의 시간 지연과 반무한 열전도 ────────────────
class PdeLaplaceTopic(EvolutionTopic):
    id = 'pde_laplace'
    title = '12.12 · Laplace 변환으로 구하는 PDE 해'
    description = '초기값 0인 반무한 현의 경계 입력 전달과 반무한 막대의 계단 온도 응답을 확인합니다.'
    inputs = (InputSpec('mode', '방정식', 'choice', '반무한 현', choices=('반무한 현', '반무한 열전도')),
              InputSpec('boundary', '경계 변위 h(t)', 'text', 'Piecewise((sin(t),t<2*pi),(0,True))', visible_in=('반무한 현',)),
              PARAMETERS, InputSpec('speed', '속력 c', 'float', 1., .01, 20, visible_in=('반무한 현',)),
              InputSpec('alpha', '열확산계수 α', 'float', .2, .000001, 20, visible_in=('반무한 열전도',)),
              InputSpec('amplitude', '계단 경계 온도 A', 'float', 1., -100, 100, visible_in=('반무한 열전도',)),
              InputSpec('extent', '표시 길이 X', 'float', 8., .1, 100),
              InputSpec('num_points', '공간 표본 수', 'int', 501, 101, 2001)) + TIME_INPUTS
    examples = {'한 주기 경계 사인파': {}, '계단 진행파': {'boundary': '1'},
                '열 침투': {'mode': '반무한 열전도'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        x = np.linspace(0, p['extent'], p['num_points'])
        time = np.linspace(0, p['end_time'], p['time_points'])
        frames = np.zeros((len(time), len(x)))
        if p['mode'] == '반무한 현':
            h = function(p['boundary'], p['parameters'], ('t',))
            delayed = time[:, None] - x / p['speed']
            active = delayed >= 0
            # 인과 영역만 평가하여 sqrt(t)처럼 음의 시간에서 정의되지 않은 입력도 처리합니다.
            frames[active] = h(delayed[active])
            symbolic = '초기 변위·속도 0\ns²U=c²U_xx\nU(x,s)=H(s) exp(−xs/c)\nu(x,t)=h(t−x/c) Heaviside(t−x/c)\nH(s)=Laplace[h(t)] (형식적 표기)'
        else:
            frames[1:] = p['amplitude'] * erfc(x[None, :] / (2*np.sqrt(p['alpha']*time[1:, None])))
            frames[0, 0] = p['amplitude']
            symbolic = '초기 온도 0, u(0,t)=A\nsU=αU_xx\nU(x,s)=(A/s) exp(−x sqrt(s/α))\nu(x,t)=A erfc(x/(2 sqrt(αt))), t>0'
        result = evolution_result(self.id, p, x, time, frames, np.zeros_like(x),
            ['x≥0의 반무한 영역이며 화면 오른쪽 끝에서 반사·고정 경계조건을 부과하지 않습니다.',
             't=0의 모서리에서 초기값과 경계값이 다르면 경계값을 표시합니다. 파면 도착 시점에는 오른쪽 값을 사용합니다.',
             '표시 공식은 이 두 문제에서 유도한 해입니다. 임의 PDE의 기호 Laplace 자동 해법은 아닙니다.'])
        result.data['symbolic'] = symbolic
        result.tables['변환 과정'] = (['단계'], [[line] for line in symbolic.splitlines()])
        return result

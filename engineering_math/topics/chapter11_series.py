"""11.2~11.6: 반구간, 강제진동, 최소제곱, 고유함수 전개 화면입니다."""
import numpy as np
from matplotlib.figure import Figure
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.expr import parameter_expression, evaluate, real_constant
from engineering_math.core.fourier import calculate_fourier_coefficients, calculate_fourier_sum
from engineering_math.core.chapter11 import half_range, project_basis, orthogonal_basis, sturm_basis, forced_response, checked


FUNCTION_INPUTS = (
    InputSpec('function', '함수 f(x)', 'text', 'x'),
    InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
)
SERIES_INPUTS = (
    InputSpec('L', '길이 L', 'constant', 'pi'),
    InputSpec('N', '차수 / 항 개수 N', 'int', 10, 1, 60),
    InputSpec('num_points', '적분 표본 수', 'int', 2001, 501, 20001),
)


def positive_length(text):
    length = real_constant(text)
    if not 1e-5 <= length <= 10000:
        raise ValueError('L은 0.00001~10000 범위의 양수여야 합니다.')
    return length


def sample(p, start, end):
    x = np.linspace(start, end, p['num_points'])
    expression = parameter_expression(p['function'], p['parameters'])
    return x, evaluate(expression, (x,))


def line_figure(x, curves, title, xlabel='x', ylabel='Value'):
    figure = Figure(figsize=(8, 5), layout='constrained')
    axis = figure.subplots()
    for label, values in curves.items():
        axis.plot(x, values, label=label)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=.3)
    axis.legend()
    return figure


class StaticTopic:
    supports_animation = False

    def figures(self, result):
        d = result.data
        return {'함수 비교': line_figure(d['x'], d['curves'], self.title.split(' · ')[0])}


# ── 11.2: 반구간 전개 ───────────────────────────────────────────
class HalfRangeTopic(StaticTopic):
    id = 'half_range'
    title = '11.2 · 반구간 사인·코사인 전개'
    description = '[0,L]에서 입력한 함수를 홀수 또는 짝수로 확장하고 2L 주기 급수와 비교합니다.'
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('mode', '확장 방식', 'choice', '사인 · 홀수 확장', choices=('사인 · 홀수 확장', '코사인 · 짝수 확장')),
    )
    examples = {'x의 사인 전개': {}, 'x의 코사인 전개': {'mode': '코사인 · 짝수 확장'},
                '상수의 사인 전개': {'function': '1'}, '포물선': {'function': 'x*(pi-x)'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        x, y = sample(p, 0, length)
        cosine = p['mode'].startswith('코사인')
        a0, coefficients, sums = half_range(x, y, length, p['N'], cosine)
        extended_x = np.linspace(-2 * length, 2 * length, 2001)
        folded = (extended_x + length) % (2 * length) - length
        expression = parameter_expression(p['function'], p['parameters'])
        extended_y = evaluate(expression, (np.abs(folded),))
        if not cosine:
            extended_y *= np.sign(folded)
        zeros = np.zeros_like(coefficients)
        extended_sum = calculate_fourier_sum(extended_x, length, a0,
                                             coefficients if cosine else zeros,
                                             zeros if cosine else coefficients, p['N'])
        error = checked(y - sums[-1])
        checked(error ** 2)
        tables = {'계수': (['n', 'a_n' if cosine else 'b_n'], list(enumerate(coefficients, 1))),
                  '수렴': (['N', 'RMSE'], [[n + 1, float(np.sqrt(np.mean((y - row) ** 2)))] for n, row in enumerate(sums)]),
                  '표본': (['x', 'f(x)', 'S_N(x)'], np.column_stack((x, y, sums[-1])))}
        data = dict(x=x, curves={'f(x)': y, 'S_N(x)': sums[-1]}, extended_x=extended_x,
                    extended_y=extended_y, extended_sum=extended_sum, coefficients=coefficients, a0=a0)
        return Result(self.id, p, {'RMSE': float(np.sqrt(np.mean(error ** 2))), 'a₀': a0}, tables, data,
                      ['교재 11.2, 12쪽. 사인 전개는 양 끝에서 0입니다. 불연속점에서는 좌우 극한의 평균에 수렴합니다.',
                       '확장 그래프의 경계 표본값과 원함수의 끝점 값은 다를 수 있습니다.'])

    def figures(self, result):
        figures = super().figures(result)
        d = result.data
        figures['주기 확장'] = line_figure(d['extended_x'], {'Periodic extension': d['extended_y'], 'Partial sum': d['extended_sum']}, 'Even / odd periodic extension')
        return figures


# ── 11.3: Fourier 외력과 진동 응답 ───────────────────────────────
class ForcedTopic(StaticTopic):
    id = 'fourier_forced'
    title = '11.3 · Fourier 외력과 강제진동'
    description = 'f(x)의 x를 시간으로 해석합니다. 한 주기 [-L,L]의 외력을 급수화하여 m y″+c y′+k y=F_N(t)를 풉니다.'
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('mass', '질량 m', 'float', 1., .01, 100),
        InputSpec('damping', '감쇠 c', 'float', .2, 0, 100),
        InputSpec('stiffness', '강성 k', 'float', 4., .01, 100),
        InputSpec('y0', '초기 변위', 'float', 0., -100, 100),
        InputSpec('v0', '초기 속도', 'float', 0., -100, 100),
        InputSpec('end_time', '마지막 시간', 'float', 20., .1, 100),
    )
    examples = {'사각파 외력': {'function': 'sign(sin(x))'},
                '무감쇠 공진': {'function': 'sin(x)', 'stiffness': 1., 'damping': 0.},
                '감쇠 공진': {'function': 'sin(x)', 'stiffness': 1., 'damping': .2}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        x, y = sample(p, -length, length)
        a0, an, bn = calculate_fourier_coefficients(x, y, length, p['N'])
        time = np.linspace(0, p['end_time'], 2001)
        state, particular, resonant, frequencies = forced_response(time, length, a0, an, bn,
            p['mass'], p['damping'], p['stiffness'], p['y0'], p['v0'])
        force = calculate_fourier_sum(time, length, a0, an, bn, p['N'])
        curves = {'Initial-value response': state[0]}
        if particular is not None:
            curves['Periodic particular solution'] = particular
        tables = {'외력 성분': (['n', 'omega', 'a_n', 'b_n', 'resonant'],
                  [[i + 1, frequencies[i], an[i], bn[i], bool(resonant[i])] for i in range(p['N'])]),
                  '응답': (['t', 'F_N', 'y', 'velocity'], np.column_stack((time, force, state.T)))}
        notices = ['교재 11.3, 22쪽. 원래 외력 대신 유한 Fourier 부분합에 대한 초기값 문제를 계산합니다.',
                   '무감쇠계의 주기 특수해는 초기값 응답의 장시간 극한을 뜻하지 않습니다.']
        if np.any(resonant):
            notices.append('활성 외력 성분에 무감쇠 공진이 있습니다. 유계 주기 특수해를 표시하지 않으며 시간 응답의 증가를 관찰하십시오.')
        return Result(self.id, p, {'최대 |y|': float(np.max(np.abs(state[0]))), '공진 성분 수': int(resonant.sum())},
                      tables, dict(x=time, curves=curves, force=force, state=state, resonant=resonant, particular=particular), notices)

    def figures(self, result):
        figures = super().figures(result)
        figures['외력'] = line_figure(result.data['x'], {'F_N(t)': result.data['force']}, 'Truncated periodic force', 't')
        return figures


# ── 11.4: 직교투영의 최소제곱 성질 ────────────────────────────────
class ApproximationTopic(StaticTopic):
    id = 'trig_approximation'
    title = '11.4 · 삼각다항식과 최소제곱 근사'
    description = 'Fourier 계수의 직교투영과 계수를 변형한 근사의 적분 제곱오차를 비교합니다.'
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('perturbation', 'a₁에 더할 값 δ', 'float', .5, -10, 10),
    )
    examples = {'x의 최소제곱 근사': {}, '절댓값': {'function': 'abs(x)'}, '사각파': {'function': 'sign(x)'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        x, y = sample(p, -length, length)
        a0, an, bn = calculate_fourier_coefficients(x, y, length, p['N'])
        approximation = calculate_fourier_sum(x, length, a0, an, bn, p['N'])
        changed = approximation + p['perturbation'] * np.cos(np.pi * x / length)
        energy = float(np.trapezoid(y ** 2, x=x))
        captured = 2 * length * a0 ** 2 + length * np.cumsum(an ** 2 + bn ** 2)
        error = float(np.trapezoid((y - approximation) ** 2, x=x))
        changed_error = float(np.trapezoid((y - changed) ** 2, x=x))
        checked([energy, error, changed_error, *captured])
        tables = {'오차와 에너지': (['N', 'captured energy', 'energy - captured'],
                    [[i + 1, captured[i], energy - captured[i]] for i in range(p['N'])]),
                  '표본': (['x', 'f', 'S_N', 'perturbed'], np.column_stack((x, y, approximation, changed)))}
        return Result(self.id, p, {'적분 제곱오차': error, '변형 후 제곱오차': changed_error,
                      '예측 오차 증가 Lδ²': length * p['perturbation'] ** 2}, tables,
                      dict(x=x, curves={'f(x)': y, 'Projection': approximation, 'Perturbed a1': changed}),
                      ['교재 11.4, 25쪽. 같은 차수의 삼각다항식 중 Fourier 투영이 연속 L² 오차를 최소화합니다.',
                       '표는 사다리꼴 적분 근사입니다. 에너지 차이는 반올림으로 작은 음수가 될 수 있습니다.'])


# ── 11.5~11.6: 고유함수와 가중 직교투영 ──────────────────────────
class OrthogonalTopic(StaticTopic):
    id = 'orthogonal_series'
    title = '11.6 · Legendre·Bessel 직교급수'
    description = 'Legendre는 [-1,1], Bessel은 [0,L]에서 계산합니다. N은 포함할 기저의 개수입니다.'
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('mode', '직교기저', 'choice', 'Legendre', choices=('Legendre', 'Bessel')),
        InputSpec('nu', 'Bessel 차수 ν', 'int', 0, 0, 5, visible_in=('Bessel',)),
    )
    examples = {'Legendre x²': {'function': 'x^2', 'N': 6},
                'Bessel 상수': {'function': '1', 'mode': 'Bessel', 'L': '1'},
                'Bessel 포물선': {'function': '1-x^2', 'mode': 'Bessel', 'L': '1'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        if p['mode'] == 'Legendre':
            x, y = sample(p, -1, 1)
        else:
            x, y = sample(p, 0, length)
        basis, weight, eigenvalues = orthogonal_basis(p['mode'], x, p['N'], length, p['nu'])
        return self.projection_result(p, x, y, basis, weight, eigenvalues,
            'Legendre: r=1, λ=n(n+1), n=0부터 시작합니다. Bessel: r=x, λ=(jν,n/L)², n=1부터 시작합니다.')

    def projection_result(self, p, x, y, basis, weight, eigenvalues, note):
        coefficients, gram, sums, norms = project_basis(x, y, basis, weight)
        errors = [float(np.trapezoid(weight * (y - row) ** 2, x=x)) for row in sums]
        checked(errors)
        gram_error = float(np.max(np.abs(gram - np.eye(len(gram)))))
        tables = {'전개 계수': (['term', 'lambda', 'coefficient', 'norm squared'],
                  [[i + 1, eigenvalues[i], coefficients[i], norms[i]] for i in range(p['N'])]),
                  '수렴': (['terms', 'weighted squared error'], list(enumerate(errors, 1))),
                  '정규화 Gram': ([str(i + 1) for i in range(p['N'])], gram),
                  '표본': (['x', 'f', 'sum'], np.column_stack((x, y, sums[-1])))}
        return Result(self.id, p, {'가중 적분 제곱오차': errors[-1], 'Gram 최대 편차': gram_error}, tables,
                      dict(x=x, curves={'f(x)': y, 'Projection': sums[-1]}, basis=basis, gram=gram,
                           coefficients=coefficients, eigenvalues=eigenvalues),
                      [note, 'Gram 행렬은 정규화된 기저의 가중 내적입니다. 단위행렬과의 차이로 적분 해상도를 점검하십시오.',
                       '표본 적분으로 계산합니다. 표본 수를 늘려 계수와 오차의 안정성을 확인하십시오.'])

    def figures(self, result):
        figures = super().figures(result)
        d = result.data
        figures['기저 함수'] = line_figure(d['x'], {f'Term {i + 1}': row for i, row in enumerate(d['basis'][:6])}, 'First six basis functions')
        figure = Figure(figsize=(6, 5), layout='constrained')
        axis = figure.subplots()
        mesh = axis.imshow(d['gram'], origin='lower', vmin=-1, vmax=1, cmap='coolwarm')
        axis.set(title='Normalized weighted Gram matrix', xlabel='Basis index (zero based)', ylabel='Basis index (zero based)')
        figure.colorbar(mesh, ax=axis)
        figures['직교성'] = figure
        return figures


class SturmTopic(OrthogonalTopic):
    id = 'sturm_liouville'
    title = '11.5 · Sturm–Liouville 고유함수'
    description = '상수계수 (p y′)′+(q+λr)y=0의 DD·NN·DN 경계조건을 비교합니다. 임의 가변계수 해석기는 아닙니다.'
    inputs = FUNCTION_INPUTS + SERIES_INPUTS + (
        InputSpec('mode', '경계조건', 'choice', 'DD', choices=('DD', 'NN', 'DN')),
        InputSpec('p', '계수 p > 0', 'float', 1., .01, 100),
        InputSpec('q', '계수 q', 'float', 0., -100, 100),
        InputSpec('r', '가중치 r > 0', 'float', 1., .01, 100),
    )
    examples = {'양 끝 고정 DD': {}, '양 끝 기울기 0 NN': {'mode': 'NN', 'function': '1'},
                '혼합 경계 DN': {'mode': 'DN'}, '고유값 이동': {'q': 2.}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        length = positive_length(p['L'])
        x, y = sample(p, 0, length)
        basis, eigenvalues = sturm_basis(x, length, p['N'], p['mode'], p['p'], p['q'], p['r'])
        return self.projection_result(p, x, y, basis, np.full_like(x, p['r']), eigenvalues,
            '교재 11.5, 29쪽의 부호: λ=(pκ²−q)/r. D는 y=0, N은 y′=0이며 DN은 왼쪽 D·오른쪽 N입니다. NN은 상수 모드를 포함합니다.')

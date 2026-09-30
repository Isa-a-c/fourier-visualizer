"""11.7~11.10의 연속 적분·변환 및 공식 검산 실험입니다."""
import numpy as np
from engineering_math.core.models import InputSpec, Result, validate_params
from engineering_math.core.expr import parameter_expression, evaluate
from engineering_math.core.chapter11 import integral_transform, checked
from .chapter11_series import StaticTopic, line_figure

ROOT_TWO_PI = np.sqrt(2 * np.pi)
ROOT_TWO_OVER_PI = np.sqrt(2 / np.pi)
TRANSFORM_INPUTS = (
    InputSpec('function', '함수 f(x)', 'text', 'exp(-x^2/2)'),
    InputSpec('parameters', '매개변수 (A=2; w=3)', 'text', ''),
    InputSpec('extent', '공간 적분 끝점 X', 'float', 8., .1, 100),
    InputSpec('cutoff', '주파수 절단 W (rad/unit)', 'float', 12., .1, 100),
    InputSpec('num_points', '공간 적분 표본 수', 'int', 2001, 501, 8193),
    InputSpec('frequency_points', '주파수 적분 표본 수', 'int', 601, 101, 1601),
)


def transform_data(p, kind):
    """무한 적분을 유한 구간으로 절단한 수치 실험입니다. 존재성 판정은 아닙니다."""
    extent, cutoff = p['extent'], p['cutoff']
    full_line = kind in ('complex', 'integral')
    x = np.linspace(-extent if full_line else 0., extent, p['num_points'])
    expression = parameter_expression(p['function'], p['parameters'])
    values = evaluate(expression, (x,))
    omega = np.linspace(-cutoff if kind == 'complex' else 0., cutoff, p['frequency_points'])
    display_x = np.linspace(x[0], x[-1], 1001)
    reference = evaluate(expression, (display_x,))
    if kind == 'integral':
        cosine = integral_transform(x, values, omega, 'cosine', 1 / np.pi)
        sine = integral_transform(x, values, omega, 'sine', 1 / np.pi)
        restored = (integral_transform(omega, cosine, display_x, 'cosine')
                    + integral_transform(omega, sine, display_x, 'sine'))
        curves = {'A(omega)': cosine, 'B(omega)': sine}
        spectrum = cosine - 1j * sine
    else:
        kernel = 'forward' if kind == 'complex' else kind
        factor = 1 / ROOT_TWO_PI if kind == 'complex' else ROOT_TWO_OVER_PI
        spectrum = integral_transform(x, values, omega, kernel, factor)
        restored = integral_transform(omega, spectrum, display_x, 'inverse' if kind == 'complex' else kind, factor)
        curves = {'Real': spectrum.real, 'Imaginary': spectrum.imag, 'Magnitude': np.abs(spectrum)}
    checked(restored)
    notices = ['공간과 주파수를 유한 구간으로 절단한 사다리꼴 적분입니다. 무한구간 변환의 존재나 정확한 역변환을 보장하지 않습니다.',
               'X·W와 각각의 표본 수를 따로 늘려 절단 오차와 적분 오차를 확인하십시오. 불연속점은 좌우 극한의 평균으로 복원됩니다.']
    if cutoff * (x[1] - x[0]) > np.pi / 4 or extent * (omega[1] - omega[0]) > np.pi / 4:
        notices.append('진동 커널의 표본이 성깁니다. 공간 또는 주파수 표본 수를 늘리십시오.')
    edges = np.abs(values[[0, -1]]) if full_line else np.abs(values[-1:])
    if np.max(edges) > .01 * max(1e-15, np.max(np.abs(values))):
        notices.append('공간 절단 경계에서 함수가 충분히 작지 않습니다. 꼬리 적분의 영향이 클 수 있습니다.')
    tables = {'변환 표본': (['omega', 'real', 'imaginary', 'magnitude'], np.column_stack((omega, spectrum.real, spectrum.imag, np.abs(spectrum)))),
              '복원 표본': (['x', 'f', 'restored real', 'restored imaginary'], np.column_stack((display_x, reference, restored.real, restored.imag)))}
    metrics = {'복원 RMSE': float(np.sqrt(np.mean(np.abs(reference - restored) ** 2))),
               '복원 허수 최대값': float(np.max(np.abs(restored.imag)))}
    checked(list(metrics.values()))
    if kind == 'integral':
        tables['Fourier 적분 계수'] = (['omega', 'A', 'B'], np.column_stack((omega, cosine, sine)))
    data = dict(x=display_x, curves={'f(x)': reference, 'Finite-band reconstruction': restored.real},
                omega=omega, spectrum=spectrum, spectrum_curves=curves, restored=restored,
                integration_x=x, integration_y=values)
    return metrics, tables, data, notices


class IntegralTopic(StaticTopic):
    id = 'fourier_integral'
    title = '11.7 · Fourier 적분'
    description = 'A(ω)·B(ω)를 적분하여 비주기 함수를 복원합니다. ω는 각주파수입니다.'
    inputs = TRANSFORM_INPUTS
    examples = {'Gaussian': {}, '사각 펄스': {'function': 'Piecewise((1, abs(x)<1), (0, True))'},
                '비대칭 펄스': {'function': 'Piecewise((1, And(x>0,x<2)), (0, True))'}}
    kind = 'integral'

    def compute(self, params):
        p = validate_params(self.inputs, params)
        metrics, tables, data, notices = transform_data(p, self.kind)
        return Result(self.id, p, metrics, tables, data, notices)

    def figures(self, result):
        figures = super().figures(result)
        figures['변환 스펙트럼'] = line_figure(result.data['omega'], result.data['spectrum_curves'], 'Continuous spectrum', 'omega (rad/unit)')
        return figures


class SineCosineTopic(IntegralTopic):
    id = 'sine_cosine_transform'
    title = '11.8 · Fourier 사인·코사인 변환'
    description = '양의 반축 함수의 변환·복원을 비교합니다. 정변환과 역변환에 모두 √(2/π)를 사용합니다.'
    inputs = TRANSFORM_INPUTS + (InputSpec('mode', '변환 종류', 'choice', '코사인', choices=('코사인', '사인')),)
    examples = {'코사인 Gaussian': {}, '사인 지수감쇠': {'mode': '사인', 'function': 'exp(-x)'},
                '사인 x Gaussian': {'mode': '사인', 'function': 'x*exp(-x^2/2)'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        metrics, tables, data, notices = transform_data(p, 'cosine' if p['mode'] == '코사인' else 'sine')
        return Result(self.id, p, metrics, tables, data, notices)


class ContinuousTransformTopic(IntegralTopic):
    id = 'continuous_fourier'
    title = '11.9 · 연속 Fourier 변환·FFT 비교'
    description = '실수 입력의 복소 스펙트럼과 역적분을 계산합니다. FFT에는 표본 간격과 원점 위상을 보정합니다.'
    kind = 'complex'
    examples = {'Gaussian 자기 변환': {}, '이동 Gaussian': {'function': 'exp(-(x-2)^2/2)'},
                '양측 지수': {'function': 'exp(-abs(x))'}}

    def compute(self, params):
        result = super().compute(params)
        p, d = result.params, result.data
        count = p['num_points']
        x = np.linspace(-p['extent'], p['extent'], count, endpoint=False)
        values = evaluate(parameter_expression(p['function'], p['parameters']), (x,))
        dx = x[1] - x[0]
        omega = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(count, d=dx))
        spectrum = np.fft.fftshift(np.fft.fft(values)) * dx / ROOT_TWO_PI * np.exp(-1j * omega * x[0])
        selected = np.flatnonzero(np.abs(omega) <= p['cutoff'])
        # 동등 주파수의 비교 표본은 최대 401개로 제한합니다.
        selected = selected[::max(1, int(np.ceil(len(selected) / 401)))]
        omega, spectrum = omega[selected], spectrum[selected]
        direct = integral_transform(d['integration_x'], d['integration_y'], omega, 'forward', 1 / ROOT_TWO_PI)
        d.update(fft_omega=omega, fft_spectrum=spectrum, direct_at_bins=direct)
        result.metrics['동일 ω에서 FFT·적분 최대 차이'] = float(np.max(np.abs(spectrum - direct)))
        result.tables['FFT 비교'] = (['omega', 'FFT real', 'FFT imag', 'integral real', 'integral imag'],
                                    np.column_stack((omega, spectrum.real, spectrum.imag, direct.real, direct.imag)))
        result.notices.append('FFT는 끝점 중복 없는 격자의 주기적 표본을 사용합니다. 적분과의 차이는 절단·구적 차이를 포함합니다. 직접 DFT 검산·aliasing은 기존 11.1 화면의 추가 분석에서 사용할 수 있습니다.')
        return result

    def figures(self, result):
        figures = super().figures(result)
        d = result.data
        figures['FFT·연속 적분 비교'] = line_figure(d['fft_omega'], {'FFT magnitude': np.abs(d['fft_spectrum']),
                          'Quadrature magnitude': np.abs(d['direct_at_bins'])}, 'Same angular-frequency bins', 'omega')
        return figures


# ── 11.10: 공식은 선택 가능한 데이터로 관리하고 수치 적분과 검산합니다 ──
PAIRS = {
    '코사인 · 지수': ('cosine', 'exp(-a*x)', 'sqrt(2/pi)*a/(a²+ω²)', 'a>0'),
    '코사인 · Gaussian': ('cosine', 'exp(-a*x^2)', 'exp(-ω²/(4a))/sqrt(2a)', 'a>0'),
    '코사인 · 사각 펄스': ('cosine', 'Piecewise((1,x<a),(0,True))', 'sqrt(2/pi)*sin(aω)/ω', 'a>0, ω=0은 극한'),
    '사인 · 지수': ('sine', 'exp(-a*x)', 'sqrt(2/pi)*ω/(a²+ω²)', 'a>0'),
    '사인 · x Gaussian': ('sine', 'x*exp(-a*x^2)', 'ω*exp(-ω²/(4a))/(2a)^(3/2)', 'a>0'),
    '사인 · 사각 펄스': ('sine', 'Piecewise((1,x<a),(0,True))', 'sqrt(2/pi)*(1-cos(aω))/ω', 'a>0, ω=0은 극한'),
    '복소 · Gaussian': ('complex', 'exp(-a*x^2)', 'exp(-ω²/(4a))/sqrt(2a)', 'a>0'),
    '복소 · 사각 펄스': ('complex', 'Piecewise((1,abs(x)<a),(0,True))', 'sqrt(2/pi)*sin(aω)/ω', 'a>0, ω=0은 극한'),
    '복소 · 유리함수': ('complex', '1/(x^2+a^2)', 'sqrt(pi/2)*exp(-a*abs(ω))/a', 'a>0'),
    '복소 · 단측 지수': ('complex', 'Piecewise((0,x<0),(1/2,Eq(x,0)),(exp(-a*x),True))', '1/(sqrt(2*pi)*(a+iω))', 'a>0, x=0은 좌우 평균'),
    '복소 · 이동 펄스': ('complex', 'Piecewise((1,And(x>b-a,x<b+a)),(0,True))', 'sqrt(2/pi)*exp(-iωb)*sin(aω)/ω', 'a>0, b는 중심'),
}


def exact_pair(name, omega, a, b=0.):
    """ω=0의 제거 가능한 특이점은 sinc와 안정적인 식으로 평가합니다."""
    if 'Gaussian' in name:
        gaussian = np.exp(-omega ** 2 / (4 * a)) / np.sqrt(2 * a)
        return gaussian * omega / (2 * a) if name.startswith('사인') else gaussian
    if name == '복소 · 유리함수':
        return np.sqrt(np.pi / 2) / a * np.exp(-a * np.abs(omega))
    if name == '복소 · 단측 지수':
        return 1 / (ROOT_TWO_PI * (a + 1j * omega))
    if '펄스' in name:
        if name.startswith('사인'):
            return ROOT_TWO_OVER_PI * a * np.sin(a * omega / 2) * np.sinc(a * omega / (2 * np.pi))
        result = ROOT_TWO_OVER_PI * a * np.sinc(a * omega / np.pi)
        return result * np.exp(-1j * omega * b) if '이동' in name else result
    return ROOT_TWO_OVER_PI * (a if name.startswith('코사인') else omega) / (a * a + omega * omega)


class TransformTableTopic(IntegralTopic):
    id = 'transform_table'
    title = '11.10 · 변환공식 표와 수치 검산'
    description = '대표 공식 11개를 선택하여 매개변수와 절단 범위를 바꾸고 수치 적분과 비교합니다.'
    inputs = (
        InputSpec('pair', '변환 공식', 'choice', '복소 · Gaussian', choices=tuple(PAIRS)),
        InputSpec('a', '매개변수 a > 0', 'float', 1., .05, 20),
        InputSpec('b', '이동 펄스 중심 b', 'float', 1., -20, 20),
    ) + TRANSFORM_INPUTS[2:]
    examples = {'Gaussian 검산': {}, '사각 펄스와 sinc': {'pair': '복소 · 사각 펄스'},
                '이동과 위상': {'pair': '복소 · 이동 펄스'}, '사인 지수': {'pair': '사인 · 지수'}}

    def compute(self, params):
        p = validate_params(self.inputs, params)
        kind, function, formula, condition = PAIRS[p['pair']]
        calculation = {key: p[key] for key in ('extent', 'cutoff', 'num_points', 'frequency_points')}
        calculation.update(function=function, parameters=f"a={p['a']}; b={p['b']}")
        metrics, tables, data, notices = transform_data(calculation, kind)
        exact = exact_pair(p['pair'], data['omega'], p['a'], p['b'])
        data['spectrum_curves'] = {'Quadrature real': data['spectrum'].real, 'Formula real': exact.real,
                                   'Quadrature imaginary': data['spectrum'].imag, 'Formula imaginary': exact.imag}
        data['exact'] = exact
        metrics['공식·적분 최대 차이'] = float(np.max(np.abs(data['spectrum'] - exact)))
        tables['공식 표'] = (['선택', '종류', 'f(x)', '변환 공식', '조건'],
                           [[name, *entry] for name, entry in PAIRS.items()])
        tables['공식 검산'] = (['omega', 'exact real', 'exact imaginary', 'absolute error'],
                              np.column_stack((data['omega'], exact.real, exact.imag, np.abs(data['spectrum'] - exact))))
        notices.insert(0, f'선택 공식: {formula}. 조건: {condition}. 교재 66~68쪽 중 대표 공식을 구현했습니다.')
        notices.append('특이 적분·조건부 수렴을 포함한 교재의 모든 표 항목을 수치 지원하는 것은 아닙니다. 공식과 절단 적분의 차이는 버그와 동일하지 않습니다.')
        return Result(self.id, p, metrics, tables, data, notices)

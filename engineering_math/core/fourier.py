"""UI 비의존 Fourier 수치 계산. 기존 v0.5.3 알고리즘을 보존합니다."""
import numpy as np
from .expr import parse_expression, evaluate, compiled, symbols_for
X = symbols_for(('x',))[0]
numpy_function = compiled

def sample_function(expression, x):
    return evaluate(expression,(x,))

def calculate_fourier_coefficients(x, y, L, max_N):
    """[-L,L] 표본에 사다리꼴 적분을 적용한다. an[0]은 a_1이다."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L은 유한한 양수여야 합니다.")
    if not isinstance(max_N, (int, np.integer)) or max_N < 1:
        raise ValueError("max_N은 양의 정수여야 합니다.")
    if x.ndim != 1 or x.size < 2 or x.shape != y.shape:
        raise ValueError("x와 y는 길이가 같은 1차원 표본이어야 합니다.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("표본은 모두 유한해야 합니다.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x는 오름차순이어야 합니다.")
    if not np.isclose(x[0], -L) or not np.isclose(x[-1], L):
        raise ValueError("표본 구간은 [-L, L]이어야 합니다.")

    # NumPy 2.x의 API를 우선 사용하고 이전 버전도 지원한다.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        a0 = float(integrate(y, x=x) / (2 * L))
        cosine_coefficients, sine_coefficients = integrate_harmonics(x,y,L,1,max_N)
    if not np.all(np.isfinite(np.r_[a0, cosine_coefficients, sine_coefficients])):
        raise ValueError("계수가 유한하지 않습니다. 함수의 크기를 줄여 주세요.")
    return a0, cosine_coefficients, sine_coefficients

def integrate_harmonics(x, y, L, first, last):
    """검증된 표본에서 필요한 차수 범위만 적분하는 내부 함수."""
    integrate = np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    an = np.empty(last-first+1)
    bn = np.empty_like(an)
    base_angle = np.pi*(x/L)
    angle = np.empty_like(x)
    wave = np.empty_like(x)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        for index,n in enumerate(range(first,last+1)):
            np.multiply(base_angle,n,out=angle)
            np.cos(angle,out=wave)
            np.multiply(y,wave,out=wave)
            an[index] = integrate(wave,x=x)/L
            np.sin(angle,out=wave)
            np.multiply(y,wave,out=wave)
            bn[index] = integrate(wave,x=x)/L
    if not np.all(np.isfinite(an)) or not np.all(np.isfinite(bn)):
        raise ValueError('계수가 유한하지 않습니다. 함수의 크기를 줄이십시오.')
    return an,bn

def calculate_fourier_sum(x, L, a0, an, bn, N):
    """a0 + Σ(an cos(nπx/L) + bn sin(nπx/L))를 계산한다."""
    return calculate_fourier_sums(x, L, a0, an, bn, [N])[N]

def calculate_fourier_sums(x, L, a0, an, bn, orders):
    """한 번 누적하여 여러 차수의 부분합을 만든다. 필요한 차수만 복사한다."""
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L은 유한한 양수여야 합니다.")
    orders = set(orders)
    if any(not isinstance(n, (int, np.integer)) or not 0 <= n <= min(len(an), len(bn)) for n in orders):
        raise ValueError("N에 필요한 Fourier 계수가 부족합니다.")
    if not orders:
        return {}
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, a0, dtype=float)
    snapshots = {0: result.copy()} if 0 in orders else {}
    with np.errstate(over="raise", invalid="raise"):
        base_angle = np.pi * (x / L)
        for n in range(1, max(orders) + 1):
            angle = n * base_angle
            result += an[n - 1] * np.cos(angle) + bn[n - 1] * np.sin(angle)
            if n in orders:
                snapshots[n] = result.copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("부분합에서 비유한 값이 발생했습니다.")
    return snapshots

class FourierSession:
    """마지막 함수/구간/표본의 계수를 보관하는 작은 데스크톱 전용 캐시."""

    def __init__(self):
        self.key = None
        self.data = None

    def prepare(self, expression_text, L, num_points, max_N):
        key = (expression_text.strip(), L, num_points)
        if key == self.key and len(self.data[4]) >= max_N:
            expression, x, y, a0, an, bn = self.data
            return expression, x, y, a0, an[:max_N], bn[:max_N]
        expression = parse_expression(key[0])
        if key == self.key:
            x, y = self.data[1:3]
            a0, previous_an, previous_bn = self.data[3:]
            extra_an, extra_bn = integrate_harmonics(x,y,L,len(previous_an)+1,max_N)
            an = np.concatenate((previous_an,extra_an))
            bn = np.concatenate((previous_bn,extra_bn))
        else:
            x = np.linspace(-L, L, num_points)
            y = sample_function(expression, x)
            a0, an, bn = calculate_fourier_coefficients(x, y, L, max_N)
        # 계산에 성공한 결과만 교체한다. 잘못된 입력은 기존 캐시를 오염시키지 않는다.
        self.key = key
        self.data = (expression, x, y, a0, an, bn)
        return self.data

def calculate_errors(y, approximation):
    """같은 표본에서의 오차: 연속 구간 전체의 최대 오차와는 구별된다."""
    with np.errstate(over="raise", invalid="raise"):
        error = y - approximation
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))
        maximum_error = float(np.max(np.abs(error)))
    return error, {"MSE": mse, "RMSE": rmse, "Maximum Absolute Error": maximum_error}

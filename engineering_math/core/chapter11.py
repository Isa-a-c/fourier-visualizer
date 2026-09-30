"""교재 11장의 UI 독립 수치 계산입니다. 적분은 유한 표본의 근사입니다."""
import numpy as np
from scipy.special import eval_legendre, jn_zeros, jv
from scipy.integrate import solve_ivp


# ── 공통 적분과 유한성 검사 ──────────────────────────────────────
def checked(values):
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise ValueError('계산 결과가 유한하지 않습니다. 입력 크기와 구간을 확인하십시오.')
    return array


def trapezoid_weights(x):
    x = checked(np.asarray(x, dtype=float))
    if x.ndim != 1 or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError('적분 격자는 증가하는 1차원 배열이어야 합니다.')
    weights = np.empty_like(x)
    weights[0] = (x[1] - x[0]) / 2
    weights[-1] = (x[-1] - x[-2]) / 2
    weights[1:-1] = (x[2:] - x[:-2]) / 2
    return weights


def project_basis(x, values, basis, weight=None):
    """직교기저의 계수, 정규화 Gram 행렬과 누적 부분합을 반환합니다."""
    values, basis = checked(values), checked(basis)
    weights = trapezoid_weights(x)
    if weight is not None:
        weights = weights * checked(weight)
    if values.shape != np.shape(x) or basis.ndim != 2 or basis.shape[1] != len(x):
        raise ValueError('함수와 기저의 표본 크기가 일치하지 않습니다.')
    norms = checked(np.sum(basis * basis * weights, axis=1))
    if np.any(norms <= 0):
        raise ValueError('기저의 제곱노름은 양수여야 합니다.')
    coefficients = checked((basis @ (weights * values)) / norms)
    gram = checked((basis * weights) @ basis.T / np.sqrt(np.outer(norms, norms)))
    sums = checked(np.cumsum(coefficients[:, None] * basis, axis=0))
    return coefficients, gram, sums, norms


def half_range(x, values, length, order, cosine):
    harmonics = np.arange(1, order + 1)
    angles = harmonics[:, None] * np.pi * x / length
    waves = np.cos(angles) if cosine else np.sin(angles)
    coefficients = 2 / length * np.trapezoid(waves * values, x=x, axis=1)
    constant = float(np.trapezoid(values, x=x) / length) if cosine else 0.
    return constant, checked(coefficients), checked(constant + np.cumsum(coefficients[:, None] * waves, axis=0))


def orthogonal_basis(kind, x, count, length=1., bessel_order=0):
    """Legendre: [-1,1], Bessel: [0,L], 고정 차수 ν의 양의 영점 사용."""
    if kind == 'Legendre':
        indices = np.arange(count)
        basis = np.array([eval_legendre(int(n), x) for n in indices])
        return basis, np.ones_like(x), indices * (indices + 1)
    roots = jn_zeros(bessel_order, count)
    basis = jv(bessel_order, roots[:, None] * x / length)
    return basis, x, (roots / length) ** 2


def sturm_basis(x, length, count, boundary, p=1., q=0., r=1.):
    """교재 부호 (p y')' + (q + λr)y=0, 상수 p,r>0의 정확한 기저."""
    if boundary == 'DD':
        frequencies = np.arange(1, count + 1) * np.pi / length
        basis = np.sin(frequencies[:, None] * x)
    elif boundary == 'NN':
        frequencies = np.arange(count) * np.pi / length
        basis = np.cos(frequencies[:, None] * x)
    elif boundary == 'DN':
        frequencies = (np.arange(count) + .5) * np.pi / length
        basis = np.sin(frequencies[:, None] * x)
    else:
        raise ValueError('지원하는 경계조건은 DD, NN, DN입니다.')
    return basis, (p * frequencies ** 2 - q) / r


# ── 주기 외력의 Fourier 성분과 운동방정식 ────────────────────────
def forced_response(time, length, a0, an, bn, mass, damping, stiffness, y0, v0):
    frequencies = np.arange(1, len(an) + 1) * np.pi / length
    step = min(.05, .2 / max(frequencies[-1], np.sqrt(stiffness / mass), damping / mass))
    if time[-1] / step > 100000:
        raise ValueError('필요한 적분 단계가 너무 많습니다. 시간·차수를 줄이거나 주기를 늘리십시오.')

    def forcing(t):
        return a0 + an @ np.cos(frequencies * t) + bn @ np.sin(frequencies * t)

    def rhs(t, state):
        return [state[1], (forcing(t) - damping * state[1] - stiffness * state[0]) / mass]

    solution = solve_ivp(rhs, (0., float(time[-1])), [y0, v0], t_eval=time,
                         method='DOP853', rtol=1e-9, atol=1e-11, max_step=step)
    if not solution.success:
        raise ValueError('진동 응답 계산이 완료되지 않았습니다: ' + solution.message)
    checked(solution.y)
    denominator = stiffness - mass * frequencies ** 2 + 1j * damping * frequencies
    amplitudes = np.hypot(an, bn)
    resonant = (np.abs(denominator) <= 1e-10 * max(stiffness, 1.)) & (amplitudes > 1e-9 * max(1., amplitudes.max()))
    particular = None
    if not np.any(resonant):
        response = np.divide(an - 1j * bn, denominator, out=np.zeros_like(denominator),
                             where=np.abs(denominator) > 1e-10 * max(stiffness, 1.))
        particular = checked(a0 / stiffness + np.real(response @ np.exp(1j * frequencies[:, None] * time)))
    return solution.y, particular, resonant, frequencies


# ── 연속 변환: 블록 처리로 큰 외적의 메모리 사용을 제한합니다 ───────
def integral_transform(grid, values, output, kernel, factor=1., block_size=64):
    values = checked(values)
    output = checked(np.asarray(output))
    if values.shape != np.shape(grid):
        raise ValueError('변환 입력 표본의 크기가 일치하지 않습니다.')
    weighted = values * trapezoid_weights(grid)
    result = np.empty(len(output), dtype=complex if kernel in ('forward', 'inverse') else float)
    for start in range(0, len(output), block_size):
        angle = output[start:start + block_size, None] * grid
        if kernel == 'forward':
            matrix = np.exp(-1j * angle)
        elif kernel == 'inverse':
            matrix = np.exp(1j * angle)
        elif kernel == 'cosine':
            matrix = np.cos(angle)
        elif kernel == 'sine':
            matrix = np.sin(angle)
        else:
            raise ValueError('지원하지 않는 변환 커널입니다.')
        result[start:start + block_size] = factor * (matrix @ weighted)
    return checked(result)

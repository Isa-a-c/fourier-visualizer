"""12장 PDE의 해석 표현과 수치 구적입니다. UI 상태를 참조하지 않습니다."""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import roots_hermite, jn_zeros, jv, eval_legendre
from .chapter11 import checked, trapezoid_weights


# ── 무한 현: 이동과 초기 속도의 누적 적분 ─────────────────────────
def dalembert(displacement, velocity, x, time, speed, quadrature_points):
    left = x[None, :] - speed * time[:, None]
    right = x[None, :] + speed * time[:, None]
    grid = np.linspace(left.min(), right.max(), quadrature_points)
    primitive = cumulative_trapezoid(velocity(grid), grid, initial=0.)
    integral = np.interp(right, grid, primitive) - np.interp(left, grid, primitive)
    frames = .5 * (displacement(left) + displacement(right)) + integral / (2 * speed)
    return checked(frames), grid


# ── 무한 막대: Fourier 해와 동등한 열핵 합성곱 ────────────────────
def infinite_heat(initial, x, time, alpha, order):
    nodes, weights = roots_hermite(order)
    frames = np.empty((len(time), len(x)))
    for i, t in enumerate(time):
        if t == 0:
            frames[i] = initial(x)
        else:
            locations = x[:, None] + 2 * np.sqrt(alpha * t) * nodes
            frames[i] = initial(locations) @ weights / np.sqrt(np.pi)
    return checked(frames)


# ── 공통 모드 시간 진화와 에너지 ─────────────────────────────────
def modal_motion(time, omega, displacement, velocity, norms):
    phase = np.outer(time, omega)
    amplitudes = np.cos(phase) * displacement + np.sin(phase) * velocity / omega
    rates = -np.sin(phase) * omega * displacement + np.cos(phase) * velocity
    energy = .5 * np.sum(norms * (rates ** 2 + (amplitudes * omega) ** 2), axis=1)
    return checked(amplitudes), checked(energy)


def rectangle_membrane(x, y, initial, velocity, length, height, speed, count, time):
    indices = np.arange(1, count + 1)
    bx = np.sin(np.outer(indices * np.pi / length, x))
    by = np.sin(np.outer(indices * np.pi / height, y))
    weights = np.outer(trapezoid_weights(x), trapezoid_weights(y))
    norm = length * height / 4
    a = bx @ (initial * weights) @ by.T / norm
    v = bx @ (velocity * weights) @ by.T / norm
    omega = speed * np.sqrt((indices[:, None] * np.pi / length) ** 2
                            + (indices[None, :] * np.pi / height) ** 2)
    amplitudes, energy = modal_motion(time, omega.ravel(), a.ravel(), v.ravel(), norm)
    frames = np.einsum('mi,tmn,nj->tij', bx, amplitudes.reshape(len(time), count, count), by, optimize=True)
    frames[:, [0, -1], :] = 0.
    frames[:, :, [0, -1]] = 0.
    return checked(frames), energy, a, v, omega


def disk_membrane(r, theta, initial, velocity, radius, speed, radial_count, angular_count, time):
    radial_weights = trapezoid_weights(r) * r
    # θ는 끝점을 중복하지 않는 주기 격자입니다.
    angular_weight = 2 * np.pi / len(theta)
    basis, norms, omega, labels = [], [], [], []
    for m in range(angular_count + 1):
        roots = jn_zeros(m, radial_count)
        for n, root in enumerate(roots, 1):
            radial = jv(m, root * r / radius)
            radial_norm = radius ** 2 / 2 * jv(m + 1, root) ** 2
            for label, angular in [('cos', np.cos(m * theta)), ('sin', np.sin(m * theta))]:
                if m == 0 and label == 'sin':
                    continue
                basis.append(np.outer(radial, angular))
                norms.append(radial_norm * (2 * np.pi if m == 0 else np.pi))
                omega.append(speed * root / radius)
                labels.append((m, n, label))
    basis, norms, omega = np.array(basis), np.array(norms), np.array(omega)
    weights = radial_weights[:, None] * angular_weight
    a = np.einsum('kij,ij->k', basis, initial * weights) / norms
    v = np.einsum('kij,ij->k', basis, velocity * weights) / norms
    amplitudes, energy = modal_motion(time, omega, a, v, norms)
    frames = (amplitudes @ basis.reshape(len(basis), -1)).reshape(len(time), len(r), len(theta))
    frames[:, -1, :] = 0.
    return checked(frames), energy, a, v, omega, labels


# ── 내부 Dirichlet 퍼텐셜: 원판(원통의 z 독립 해), 축대칭 구 ────────
def disk_potential(boundary, radius_grid, theta, radius, count):
    values = boundary(theta)
    constant = float(np.mean(values))
    coefficients = []
    field = np.full((len(radius_grid), len(theta)), constant)
    for n in range(1, count + 1):
        cosine = 2 * np.mean(values * np.cos(n * theta))
        sine = 2 * np.mean(values * np.sin(n * theta))
        field += np.outer((radius_grid / radius) ** n, cosine * np.cos(n * theta) + sine * np.sin(n * theta))
        coefficients.append((n, cosine, sine))
    return checked(field), constant, coefficients


def sphere_potential(boundary, radius_grid, theta, radius, count, quadrature_points):
    mu = np.linspace(-1, 1, quadrature_points)
    values = boundary(mu)
    field = np.zeros((len(radius_grid), len(theta)))
    coefficients = []
    for n in range(count):
        coefficient = (2 * n + 1) / 2 * np.trapezoid(values * eval_legendre(n, mu), x=mu)
        field += coefficient * np.outer((radius_grid / radius) ** n, eval_legendre(n, np.cos(theta)))
        coefficients.append(coefficient)
    return checked(field), np.array(coefficients)

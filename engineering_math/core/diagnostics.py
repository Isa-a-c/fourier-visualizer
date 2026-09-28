"""수치 안정성 비교와 같은 평가 격자의 차수별 오차."""
import numpy as np
from engineering_math.core.fourier import sample_function, calculate_fourier_coefficients, calculate_errors


def integration_stability(expression, L, count, N):
    """M과 2M-1 표본의 사다리꼴 적분을 비교한다. 엄밀한 오차 경계는 아니다."""
    grids = [np.linspace(-L, L, size) for size in (count, 2*count-1)]
    coefficients = [calculate_fourier_coefficients(grid, sample_function(expression, grid), L, N)
                    for grid in grids]
    coarse, fine = [np.r_[a0, an, bn] for a0, an, bn in coefficients]
    difference = np.abs(fine-coarse)
    scale = float(np.max(np.abs(fine)))
    relative = float(np.max(difference)/scale) if scale > 1e-12 else np.nan
    return coefficients, difference, relative


def convergence_errors(x, y, L, a0, an, bn):
    """동일한 계수와 평가 표본으로 N=1..len(an)의 MSE/RMSE를 비교한다."""
    result = np.full_like(x, a0, dtype=float)
    angle = np.pi*x/L
    rows = []
    with np.errstate(over='raise', invalid='raise'):
        for n, (a, b) in enumerate(zip(an,bn),1):
            result += a*np.cos(n*angle)+b*np.sin(n*angle)
            _, metrics = calculate_errors(y, result)
            rows.append((n,metrics['MSE'],metrics['RMSE']))
    return np.asarray(rows)

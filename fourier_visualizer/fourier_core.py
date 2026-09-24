"""Fourier 급수의 입력 해석, 수치 계산 및 그래프 생성 공통 모듈."""

import ast

import re

import matplotlib.pyplot as plt

import numpy as np

from functools import lru_cache

import sympy as sp

X = sp.Symbol("x", real=True)

ALLOWED_NAMES = {
    "x": X, "pi": sp.pi, "E": sp.E,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "Abs": sp.Abs, "sign": sp.sign,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
}

ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
    ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.UAdd, ast.USub,
)

@lru_cache(maxsize=32)
def parse_expression(expression_text):
    """허용 목록을 검사한 문자열을 SymPy 수식으로 변환한다."""
    if not expression_text.strip() or len(expression_text) > 300:
        raise ValueError("수식을 1~300자 이내로 입력하세요.")
    # 이름, 숫자, 괄호, 기본 연산자 외에는 허용하지 않는다.
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s]+", expression_text):
        raise ValueError("지원하지 않는 문자가 있습니다. 거듭제곱은 **로 입력하세요.")
    tree = ast.parse(expression_text, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError("기본 산술 연산과 지원 함수만 사용할 수 있습니다.")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise ValueError(f"지원하지 않는 이름: {node.id}")
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("숫자 상수만 사용할 수 있습니다.")
        if isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name)
                    or node.func.id not in ALLOWED_NAMES
                    or not callable(ALLOWED_NAMES[node.func.id])
                    or len(node.args) != 1 or node.keywords):
                raise ValueError("지원 함수에는 인자 하나만 입력하세요.")
    expression = sp.sympify(expression_text, locals=ALLOWED_NAMES)
    if not isinstance(expression, sp.Expr) or expression.free_symbols - {X}:
        raise ValueError("x에 대한 실수 함수를 입력하세요.")
    return expression

def sample_function(expression, x):
    """상수 함수도 x와 같은 크기로 확장하고 비유한/복소 값을 거부한다."""
    numpy_function = sp.lambdify(X, expression, modules="numpy")
    with np.errstate(all="ignore"):
        values = np.asarray(numpy_function(x))
    if np.iscomplexobj(values):
        raise ValueError("복소수 값이 발생했습니다. 실수 함수를 입력하세요.")
    values = np.array(np.broadcast_to(values, x.shape), dtype=float, copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError("샘플에서 NaN 또는 inf가 발생했습니다. 함수와 구간을 확인하세요.")
    return values

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
    cosine_coefficients = np.zeros(max_N)
    sine_coefficients = np.zeros(max_N)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        a0 = float(integrate(y, x=x) / (2 * L))
        base_angle = np.pi * (x / L)
        for n in range(1, max_N + 1):
            angle = n * base_angle
            cosine_coefficients[n - 1] = integrate(y * np.cos(angle), x=x) / L
            sine_coefficients[n - 1] = integrate(y * np.sin(angle), x=x) / L
    if not np.all(np.isfinite(np.r_[a0, cosine_coefficients, sine_coefficients])):
        raise ValueError("계수가 유한하지 않습니다. 함수의 크기를 줄여 주세요.")
    return a0, cosine_coefficients, sine_coefficients

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

def make_comparison_figure(x, y, approximations, title):
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(x, y, color="black", linewidth=2, label="f(x)")
    for order, approximation in approximations.items():
        axes.plot(x, approximation, linewidth=1.4, label=f"N = {order}")
    axes.set(title=title, xlabel="x", ylabel="Function value")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure

def make_error_figure(x, error):
    figure, axes = plt.subplots(figsize=(12, 3))
    axes.plot(x, error, label="f(x) - S_N(x)", color="tab:red")
    axes.axhline(0, color="gray", linewidth=0.8)
    axes.set(title="Approximation error", xlabel="x", ylabel="Error")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure

def make_spectrum_figure(an, bn):
    orders = np.arange(1, len(an) + 1)
    figure, axes = plt.subplots(figsize=(10, 3.5))
    axes.stem(orders, np.abs(an), linefmt="C0-", markerfmt="C0o",
              basefmt=" ", label="|a_n|")
    axes.stem(orders, np.abs(bn), linefmt="C1--", markerfmt="C1x",
              basefmt=" ", label="|b_n|")
    axes.set(title="Coefficient spectrum", xlabel="n", ylabel="Coefficient magnitude")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure

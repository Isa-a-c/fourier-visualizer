"""선택적으로 실행하는 Streamlit 화면. 공통 계산은 fourier_core에 있다."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from fourier_core import (
    parse_expression, sample_function, calculate_fourier_coefficients,
    calculate_fourier_sum, calculate_fourier_sums, calculate_errors,
    make_comparison_figure, make_error_figure, make_spectrum_figure,
)


def show_figure(figure):
    """표시 후 Figure를 닫아 Streamlit 재실행 시 메모리 누적을 막는다."""
    try:
        st.pyplot(figure)
    finally:
        plt.close(figure)

def main():
    st.set_page_config(page_title="Fourier Series Visualizer v0.2", layout="wide")
    st.title("Fourier Series Visualizer v0.2")
    st.write("[-L, L]에서 정의된 함수의 주기적 확장과 Fourier 부분합을 살펴봅니다.")
    st.latex(r"f(x) \sim a_0 + \sum_{n=1}^{\infty}\left[a_n\cos\frac{n\pi x}{L}+b_n\sin\frac{n\pi x}{L}\right]")

    with st.sidebar:
        st.header("함수 및 계산 설정")
        expression_text = st.text_input("f(x)", value="x")
        st.caption("예: x, x**2, sin(x), cos(x), exp(x), abs(x), sign(x), 3")
        length_text = st.text_input("L (양수, 기본값 pi)", value="pi")
        current_N = st.slider("현재 Fourier 차수 N", 1, 100, 10)
        num_points = st.slider("샘플링 점 개수", 500, 20000, 5000, step=100)
        comparison_N = st.multiselect("비교할 N", [1, 3, 5, 10, 20, 50], default=[1, 3, 5, 10])

    try:
        expression = parse_expression(expression_text)
        length_expression = parse_expression(length_text)
        if length_expression.free_symbols:
            raise ValueError("L에는 x가 없는 양수 상수를 입력하세요.")
        L = float(length_expression)
        if not np.isfinite(L) or L <= 0 or not np.isfinite(2 * L):
            raise ValueError("L 및 주기 2L은 유한한 양수여야 합니다.")
        x = np.linspace(-L, L, num_points)
        y = sample_function(expression, x)
        # 빈 비교 목록도 안전하고 현재 N이 더 커도 충분히 계산한다.
        max_N = max([current_N, *comparison_N])
        a0, an, bn = calculate_fourier_coefficients(x, y, L, max_N)
        orders = sorted(set([current_N, *comparison_N]))
        approximations = calculate_fourier_sums(x, L, a0, an, bn, orders)
        error, metrics = calculate_errors(y, approximations[current_N])
        convergence_rows = []
        for order in sorted(comparison_N):
            _, order_metrics = calculate_errors(y, approximations[order])
            convergence_rows.append({"N": order, "MSE": order_metrics["MSE"], "RMSE": order_metrics["RMSE"]})
    except Exception as exc:
        st.error(f"입력 또는 수치 계산을 확인하세요: {exc}")
        return

    left, right = st.columns(2)
    with left:
        st.subheader(f"현재 N = {current_N}의 Fourier approximation")
        show_figure(make_comparison_figure(x, y, {current_N: approximations[current_N]}, "Current approximation"))
    with right:
        st.subheader("여러 N의 convergence comparison")
        if not comparison_N:
            st.info("비교할 N을 선택하면 부분합이 추가됩니다.")
        show_figure(make_comparison_figure(x, y, {n: approximations[n] for n in sorted(comparison_N)}, "Convergence comparison"))

    for column, (name, value) in zip(st.columns(3), metrics.items()):
        column.metric(name, f"{value:.6g}")
    st.caption("오차는 양 끝점을 포함한 표본에서 계산합니다. 불연속점의 값과 주기 경계 때문에 최대 오차가 0으로 수렴하지 않을 수 있습니다.")
    st.subheader("Error graph")
    show_figure(make_error_figure(x, error))

    st.subheader("Fourier coefficient table")
    st.write(f"a0 = {a0:.12g}")
    st.caption(f"현재 N과 비교 N에 필요한 계수 n = 1, …, {max_N}을 표시합니다.")
    coefficient_table = pd.DataFrame({
        "n": np.arange(1, max_N + 1), "a_n": an, "b_n": bn,
        "|a_n|": np.abs(an), "|b_n|": np.abs(bn),
    })
    st.dataframe(coefficient_table, hide_index=True)
    st.subheader("Coefficient spectrum")
    show_figure(make_spectrum_figure(an, bn))

    st.subheader("N별 수렴 오차 비교")
    st.dataframe(pd.DataFrame(convergence_rows, columns=["N", "MSE", "RMSE"]), hide_index=True)
    st.caption("sign(x)를 입력하고 N을 증가시키면 x = 0 및 주기 경계 근처의 Gibbs 진동을 관찰할 수 있습니다. 진동 영역은 좁아지지만 overshoot는 남습니다.")

    st.divider()
    st.subheader("현재 함수 정보")
    st.latex("f(x) = " + sp.latex(expression))
    st.write(f"구간: [{-L:.8g}, {L:.8g}] · period = 2L = {2 * L:.8g} · current N = {current_N}")

if __name__ == "__main__":
    main()

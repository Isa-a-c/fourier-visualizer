"""결과 데이터만 받는 공통 1D 그래프 함수."""
import numpy as np

def make_comparison_figure(x, y, approximations, title):
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(12, 3))
    axes.plot(x, error, label="f(x) - S_N(x)", color="tab:red")
    axes.axhline(0, color="gray", linewidth=0.8)
    axes.set(title="Approximation error", xlabel="x", ylabel="Error")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure

def make_spectrum_figure(an, bn):
    import matplotlib.pyplot as plt
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

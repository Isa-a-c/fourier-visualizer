"""브라우저 없이 실행되는 Tkinter Fourier 학습 프로그램."""

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

from fourier_core import (
    parse_expression, FourierSession, calculate_fourier_sums,
    calculate_errors, make_comparison_figure,
    make_error_figure, make_spectrum_figure,
)


class FourierDesktop(tk.Tk):
    """입력 상태와 화면을 관리하며 계산은 fourier_core에 위임한다."""

    def __init__(self):
        super().__init__()
        self.title("Fourier Series Visualizer v0.2 — Desktop")
        self.geometry("1400x900")
        self.minsize(1050, 700)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.figures = []
        self.session = FourierSession()
        self.pending_update = None
        self.expression = tk.StringVar(value="x")
        self.length = tk.StringVar(value="pi")
        self.order = tk.IntVar(value=10)
        self.order_text = tk.StringVar(value="10")
        self.points = tk.IntVar(value=5000)
        self.points_text = tk.StringVar(value="5000")
        self.status = tk.StringVar(value="")
        self.information = tk.StringVar(value="")
        self.comparisons = {n: tk.BooleanVar(value=n in [1, 3, 5, 10]) for n in [1, 3, 5, 10, 20, 50]}
        self.metric_values = [tk.StringVar(value="—") for _ in range(3)]
        self.build_ui()
        self.order.trace_add("write", self.sync_order_input)
        self.order_text.trace_add("write", self.apply_order_input)
        self.points.trace_add("write", self.sync_points_input)
        self.points_text.trace_add("write", self.apply_points_input)
        self.bind("<Return>", lambda event: self.recalculate())
        self.after(100, self.recalculate)

    # ── 입력 패널과 결과 탭 ───────────────────────────────────────────
    def build_ui(self):
        sidebar = ttk.Frame(self, padding=16, width=245)
        sidebar.pack(side="left", fill="y")
        ttk.Label(sidebar, text="Fourier Series", font=("맑은 고딕", 17, "bold")).pack(anchor="w")
        ttk.Label(sidebar, text="데스크톱 시각화 v0.2").pack(anchor="w", pady=(0, 20))
        for label, variable in [("함수 f(x)", self.expression), ("구간의 반길이 L", self.length)]:
            ttk.Label(sidebar, text=label).pack(anchor="w", pady=(10, 3))
            ttk.Entry(sidebar, textvariable=variable, width=25).pack(fill="x")
        ttk.Label(sidebar, text="x, x**2, sin(x), cos(x)\nexp(x), abs(x), sign(x), 3").pack(anchor="w", pady=10)
        ttk.Label(sidebar, text="현재 Fourier 차수 N").pack(anchor="w")
        # 편집 중 빈 문자열은 허용하고, 실제 차수는 1~100의 정수로 유지한다.
        self.order_input = ttk.Spinbox(
            sidebar, from_=1, to=100, increment=1, width=8,
            textvariable=self.order_text, validate="key",
            validatecommand=(self.register(self.validate_order_input), "%P"),
        )
        self.order_input.pack(anchor="w", pady=(4, 0))
        self.order_input.bind("<FocusOut>", self.finish_order_input)
        tk.Scale(sidebar, from_=1, to=100, orient="horizontal", variable=self.order,
                 command=self.schedule_update).pack(fill="x")
        ttk.Label(sidebar, text="샘플링 점 개수").pack(anchor="w", pady=(10, 0))
        self.points_input = ttk.Spinbox(
            sidebar, from_=500, to=20000, increment=1, width=8,
            textvariable=self.points_text, validate="key",
            validatecommand=(self.register(self.validate_points_input), "%P"),
        )
        self.points_input.pack(anchor="w", pady=(4, 0))
        self.points_input.bind("<FocusOut>", self.finish_points_input)
        tk.Scale(sidebar, from_=500, to=20000, resolution=1, orient="horizontal",
                 variable=self.points, command=self.schedule_update).pack(fill="x")
        ttk.Label(sidebar, text="비교할 N (모두 해제 가능)").pack(anchor="w", pady=(15, 5))
        for n, variable in self.comparisons.items():
            ttk.Checkbutton(sidebar, text=f"N = {n}", variable=variable,
                            command=self.schedule_update).pack(anchor="w")
        ttk.Button(sidebar, text="계산 / 그래프 갱신", command=self.recalculate).pack(fill="x", pady=20)
        ttk.Label(sidebar, text="함수와 L 변경 후 Enter 또는\n계산 버튼을 누르세요.\n\nsign(x)에서 N을 늘려\nGibbs 현상을 확인하세요.").pack(anchor="w")

        content = ttk.Frame(self, padding=12)
        content.pack(side="left", fill="both", expand=True)
        ttk.Label(content, text="f(x) ≈ a₀ + Σ [aₙ cos(nπx/L) + bₙ sin(nπx/L)]",
                  font=("맑은 고딕", 13)).pack(anchor="w", pady=(0, 8))
        metric_frame = ttk.Frame(content)
        metric_frame.pack(fill="x", pady=8)
        for index, label in enumerate(["MSE", "RMSE", "Maximum Absolute Error"]):
            frame = ttk.LabelFrame(metric_frame, text=label, padding=10)
            frame.pack(side="left", fill="x", expand=True, padx=4)
            ttk.Label(frame, textvariable=self.metric_values[index], font=("Segoe UI", 17)).pack()
        tk.Label(content, textvariable=self.status, fg="#b42318", anchor="w", wraplength=850).pack(fill="x", pady=5)
        self.tabs = ttk.Notebook(content)
        self.tabs.pack(fill="both", expand=True)
        self.plot_tab = ttk.Frame(self.tabs)
        self.error_tab = ttk.Frame(self.tabs)
        self.coefficient_tab = ttk.Frame(self.tabs)
        self.convergence_tab = ttk.Frame(self.tabs)
        for frame, title in [(self.plot_tab, "함수 / 부분합 비교"), (self.error_tab, "오차 그래프"),
                             (self.coefficient_tab, "계수 표 / Spectrum"), (self.convergence_tab, "N별 수렴 오차")]:
            self.tabs.add(frame, text=title)
        ttk.Label(content, textvariable=self.information, wraplength=900).pack(anchor="w", pady=10)
        ttk.Label(content, text="오차는 표본 기준입니다. 불연속점과 주기 경계에서 최대 오차가 0으로 수렴하지 않을 수 있습니다.").pack(anchor="w")

    @staticmethod
    def validate_order_input(text):
        return text == "" or (text.isascii() and text.isdigit() and 1 <= int(text) <= 100)

    def apply_order_input(self, *_):
        text = self.order_text.get()
        if text and self.validate_order_input(text):
            value = int(text)
            if self.order.get() != value:
                self.order.set(value)
                self.schedule_update()

    def sync_order_input(self, *_):
        value = str(self.order.get())
        if self.order_text.get() != value:
            self.order_text.set(value)

    def finish_order_input(self, *_):
        self.sync_order_input()

    def schedule_update(self, *_):
        if self.pending_update is not None:
            self.after_cancel(self.pending_update)
        self.pending_update = self.after(250, self.recalculate)

    @staticmethod
    def validate_points_input(text):
        # 500을 입력하는 중의 '5', '50'도 허용하되 계산에는 적용하지 않는다.
        return text == "" or (
            text.isascii() and text.isdigit() and len(text) <= 5 and int(text) <= 20000
        )

    def apply_points_input(self, *_):
        text = self.points_text.get()
        if text and self.validate_points_input(text) and 500 <= int(text) <= 20000:
            value = int(text)
            if self.points.get() != value:
                self.points.set(value)
                self.schedule_update()

    def sync_points_input(self, *_):
        value = str(self.points.get())
        if self.points_text.get() != value:
            self.points_text.set(value)

    def finish_points_input(self, *_):
        # 빈 값 또는 500 미만으로 편집을 마치면 마지막 유효값으로 복원한다.
        self.sync_points_input()

    # ── 그래프 및 스크롤 가능한 표 공통 처리 ──────────────────────────
    def embed_figure(self, parent, figure):
        self.figures.append(figure)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    @staticmethod
    def table(parent, columns, rows):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=9)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=120, anchor="center")
        for row in rows:
            tree.insert("", "end", values=[f"{v:.8g}" if isinstance(v, (float, np.floating)) else v for v in row])
        scrollbar.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

    def clear_results(self):
        for tab in [self.plot_tab, self.error_tab, self.coefficient_tab, self.convergence_tab]:
            for widget in tab.winfo_children():
                widget.destroy()
        for figure in self.figures:
            plt.close(figure)
        self.figures.clear()

    # ── 수치 계산과 화면 갱신 ─────────────────────────────────────────
    def recalculate(self):
        self.finish_order_input()
        self.finish_points_input()
        if self.pending_update is not None:
            self.after_cancel(self.pending_update)
            self.pending_update = None
        try:
            L = float(parse_expression(self.length.get()))
            if not np.isfinite(L) or L <= 0 or not np.isfinite(2 * L):
                raise ValueError("L과 주기 2L은 유한한 양수여야 합니다.")
            N = self.order.get()
            selected = [n for n, value in self.comparisons.items() if value.get()]
            expression, x, y, a0, an, bn = self.session.prepare(
                self.expression.get(), L, self.points.get(), max([N, *selected])
            )
            sums = calculate_fourier_sums(x, L, a0, an, bn, [N, *selected])
            error, metrics = calculate_errors(y, sums[N])
            convergence = []
            for n in selected:
                _, values = calculate_errors(y, sums[n])
                convergence.append([n, values["MSE"], values["RMSE"]])
        except Exception as exc:
            self.status.set(f"입력 오류: {exc}")
            self.clear_results()
            for value in self.metric_values:
                value.set("—")
            self.information.set("함수와 L을 수정한 뒤 다시 계산하세요.")
            return

        self.status.set("")
        self.clear_results()
        for variable, value in zip(self.metric_values, metrics.values()):
            variable.set(f"{value:.6g}")
        self.plot_tab.columnconfigure((0, 1), weight=1, uniform="charts")
        self.plot_tab.rowconfigure(0, weight=1)
        for index, (title, data) in enumerate([
            (f"Current approximation: N={N}", {N: sums[N]}),
            ("Convergence comparison", {n: sums[n] for n in selected}),
        ]):
            frame = ttk.Frame(self.plot_tab)
            frame.grid(row=0, column=index, sticky="nsew")
            self.embed_figure(frame, make_comparison_figure(x, y, data, title))
        self.embed_figure(self.error_tab, make_error_figure(x, error))
        ttk.Label(self.coefficient_tab, text=f"a₀ = {a0:.12g}", padding=8).pack(anchor="w")
        spectrum = ttk.Frame(self.coefficient_tab, height=280)
        spectrum.pack(fill="both", expand=True)
        self.embed_figure(spectrum, make_spectrum_figure(an, bn))
        self.table(self.coefficient_tab, ["n", "a_n", "b_n", "|a_n|", "|b_n|"],
                   [[n, a, b, abs(a), abs(b)] for n, (a, b) in enumerate(zip(an, bn), start=1)])
        self.table(self.convergence_tab, ["N", "MSE", "RMSE"], convergence)
        self.information.set(f"f(x) = {expression}    구간: [{-L:.6g}, {L:.6g}]    period = 2L = {2*L:.6g}    current N = {N}")

    def close(self):
        if self.pending_update is not None:
            self.after_cancel(self.pending_update)
        self.clear_results()
        self.destroy()


if __name__ == "__main__":
    FourierDesktop().mainloop()

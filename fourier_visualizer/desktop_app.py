"""브라우저 없이 실행되는 Tkinter Fourier 학습 프로그램."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from pathlib import Path
from project_io import validate_settings, export_results, export_transform
from animation import AnimationWindow
from background import BackgroundJob, calculate_request
from version import VERSION, build_label
from advanced_panel import AdvancedPanel
from advanced import symmetry

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
        self.title(f"Fourier Visualizer v{VERSION} · {build_label()}")
        self.geometry("1400x900")
        self.minsize(1050, 700)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.figures = []
        self.tables = {}
        self.last_result = None
        self.last_settings = None
        self.animations = []
        self.session = FourierSession()
        self.pending_update = None
        self.expression = tk.StringVar(value="x")
        self.example = tk.StringVar(value="직접 입력")
        self.length = tk.StringVar(value="pi")
        self.order = tk.IntVar(value=10)
        self.order_text = tk.StringVar(value="10")
        self.points = tk.IntVar(value=5000)
        self.points_text = tk.StringVar(value="5000")
        self.status = tk.StringVar(value="")
        self.task_text = tk.StringVar(value="")
        self.information = tk.StringVar(value="")
        self.comparisons = {n: tk.BooleanVar(value=n in [1, 3, 5, 10]) for n in [1, 3, 5, 10, 20, 50]}
        self.metric_values = [tk.StringVar(value="—") for _ in range(3)]
        self.build_ui()
        self.calculate_job = BackgroundJob(self, self.task_state)
        menu = tk.Menu(self)
        files = tk.Menu(menu, tearoff=False)
        files.add_command(label='설정 저장 (JSON)', command=self.save_settings)
        files.add_command(label='설정 불러오기', command=self.load_settings)
        files.add_command(label='결과 CSV 묶음 저장 (ZIP)', command=self.save_results)
        files.add_command(label='DFT 결과 CSV 묶음 저장 (ZIP)', command=self.save_transform)
        menu.add_cascade(label='파일', menu=files)
        menu.add_command(label='부분합 애니메이션',command=self.request_animation)
        self.config(menu=menu)
        self.order.trace_add("write", self.sync_order_input)
        self.order_text.trace_add("write", self.apply_order_input)
        self.points.trace_add("write", self.sync_points_input)
        self.points_text.trace_add("write", self.apply_points_input)
        self.bind("<Return>", lambda event: self.request_calculation())
        self.pending_update = self.after(100, self.request_calculation)

    # ── 입력 패널과 결과 탭 ───────────────────────────────────────────
    def build_ui(self):
        sidebar = ttk.Frame(self, padding=16, width=245)
        sidebar.pack(side="left", fill="y")
        ttk.Label(sidebar, text="Fourier Series", font=("맑은 고딕", 17, "bold")).pack(anchor="w")
        ttk.Label(sidebar, text=f"급수 · DFT · FFT v{VERSION}").pack(anchor="w", pady=(0, 20))
        ttk.Label(sidebar, text="예제 선택 (L = pi로 적용)").pack(anchor="w")
        example_box = ttk.Combobox(
            sidebar, textvariable=self.example, state="readonly", width=23,
            values=["직접 입력", "x", "x**2", "sin(x)", "cos(x)", "exp(x)", "abs(x)", "sign(x)", "3"],
        )
        example_box.pack(fill="x", pady=4)
        example_box.bind("<<ComboboxSelected>>", self.select_example)
        for label, variable in [("함수 f(x)", self.expression), ("구간의 반길이 L", self.length)]:
            ttk.Label(sidebar, text=label).pack(anchor="w", pady=(10, 3))
            ttk.Entry(sidebar, textvariable=variable, width=25).pack(fill="x")
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
        ttk.Label(sidebar, text="급수 적분용 표본 수").pack(anchor="w", pady=(10, 0))
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
        comparison_box = ttk.Frame(sidebar)
        comparison_box.pack(fill="x")
        for index, (n, variable) in enumerate(self.comparisons.items()):
            ttk.Checkbutton(comparison_box, text=f"N = {n}", variable=variable,
                            command=self.schedule_update).grid(row=index // 2, column=index % 2, sticky="w", padx=4)
        ttk.Button(sidebar, text="계산 / 그래프 갱신", command=self.request_calculation).pack(fill="x", pady=(12, 4))
        ttk.Button(sidebar, text="기본값으로 초기화", command=lambda:self.reset_defaults(asynchronous=True)).pack(fill="x", pady=(0, 10))
        ttk.Label(sidebar, text="함수와 L 변경 후 Enter로 적용\n그래프 저장: 아래 툴바의 디스크 아이콘").pack(anchor="w")

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
        task_frame = ttk.Frame(content)
        task_frame.pack(fill='x')
        self.main_progress = ttk.Progressbar(task_frame, mode='indeterminate', length=100)
        self.main_progress.pack(side='left')
        ttk.Label(task_frame,textvariable=self.task_text).pack(side='left',padx=8)
        ttk.Button(task_frame,text='계산 취소',command=self.cancel_calculation).pack(side='right')
        self.tabs = ttk.Notebook(content)
        self.tabs.pack(fill="both", expand=True)
        self.plot_tab = ttk.Frame(self.tabs)
        self.error_tab = ttk.Frame(self.tabs)
        self.coefficient_tab = ttk.Frame(self.tabs)
        self.convergence_tab = ttk.Frame(self.tabs)
        for frame, title in [(self.plot_tab, "함수 / 부분합 비교"), (self.error_tab, "오차 그래프"),
                             (self.coefficient_tab, "계수 표 / Spectrum"), (self.convergence_tab, "N별 수렴 오차")]:
            self.tabs.add(frame, text=title)
        self.advanced_panel = AdvancedPanel(self.tabs)
        self.tabs.add(self.advanced_panel, text="확장 실험 v0.4")
        self.tabs.bind('<<NotebookTabChanged>>', self.show_advanced)
        ttk.Label(content, textvariable=self.information, wraplength=900).pack(anchor="w", pady=10)
        ttk.Label(content, text="오차는 표본 기준입니다. 불연속점과 주기 경계에서 최대 오차가 0으로 수렴하지 않을 수 있습니다.").pack(anchor="w")

    def show_advanced(self, *_):
        if self.tabs.select() == str(self.advanced_panel):
            self.advanced_panel.request_render()

    def select_example(self, *_):
        """예제는 함수와 구간만 바꾸고 차수와 표본 수는 유지한다."""
        if self.example.get() != "직접 입력":
            self.expression.set(self.example.get())
            self.length.set("pi")
            self.recalculate(asynchronous=bool(_))

    def reset_defaults(self, asynchronous=False):
        """오류 상태에서도 한 번에 기본 실험으로 돌아간다."""
        self.example.set("직접 입력")
        self.expression.set("x")
        self.length.set("pi")
        self.order.set(10)
        self.points.set(5000)
        for n, variable in self.comparisons.items():
            variable.set(n in [1, 3, 5, 10])
        self.tabs.select(self.plot_tab)
        for variable, value in [(self.advanced_panel.M, '64'), (self.advanced_panel.frequency, '9'),
                                (self.advanced_panel.rate, '12'), (self.advanced_panel.center, '0'),
                                (self.advanced_panel.width, '0.5'), (self.advanced_panel.orders, '1,10,50'),
                                (self.advanced_panel.mode, 'Gibbs 확대')]:
            variable.set(value)
        self.recalculate(asynchronous=asynchronous)

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
        self.pending_update = self.after(250, self.request_calculation)

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

    def table(self, parent, columns, rows):
        key = tuple(columns)
        if key in self.tables:
            tree = self.tables[key]
            tree.delete(*tree.get_children())
            for row in rows:
                tree.insert('', 'end', values=[f'{v:.8g}' if isinstance(v, (float, np.floating)) else v for v in row])
            return
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
        self.tables[key] = tree

    def clear_results(self):
        self.advanced_panel.set_data(None)
        for tab in [self.plot_tab, self.error_tab, self.coefficient_tab, self.convergence_tab]:
            for widget in tab.winfo_children():
                widget.destroy()
        for figure in self.figures:
            plt.close(figure)
        self.figures.clear()
        self.tables.clear()
        self.last_result = None
        self.last_settings = None

    # ── 수치 계산과 화면 갱신 ─────────────────────────────────────────
    def task_state(self, busy, text):
        self.task_text.set(text)
        if busy: self.main_progress.start(12)
        else: self.main_progress.stop()

    def cancel_calculation(self):
        if self.pending_update is not None:
            self.after_cancel(self.pending_update)
            self.pending_update = None
        self.calculate_job.cancel()
        self.advanced_panel.cancel()

    def request_calculation(self):
        self.recalculate(asynchronous=True)

    def recalculate(self, asynchronous=False):
        self.main_progress.stop()
        if self.example.get() != '직접 입력' and (
                self.expression.get() != self.example.get() or self.length.get().strip() != 'pi'):
            self.example.set('직접 입력')
        self.finish_order_input()
        self.finish_points_input()
        if self.pending_update is not None:
            self.after_cancel(self.pending_update)
            self.pending_update = None
        self.calculate_job.cancel(notify=False)
        self.advanced_panel.cancel(notify=False)
        settings = self.current_settings()
        payload = (settings,(self.session.key,self.session.data))
        if asynchronous:
            self.calculate_job.start('series',payload,
                lambda result:self.apply_calculation(result,settings),self.calculation_error)
        else:
            try: self.apply_calculation(calculate_request('series',payload),settings)
            except Exception as exc: self.calculation_error(str(exc))

    def calculation_error(self, message):
        self.status.set(f'입력 오류: {message}')
        self.clear_results()
        for value in self.metric_values: value.set('—')
        self.information.set('함수와 L을 수정한 뒤 다시 계산하십시오.')

    def apply_calculation(self, result, settings):
        expression,x,y,L,N,selected,a0,an,bn,sums,error,metrics,convergence,cache = result
        self.session.key,self.session.data = cache
        self.status.set("")
        for variable, value in zip(self.metric_values, metrics.values()):
            variable.set(f"{value:.6g}")
        if self.figures:
            self.refresh_plots(x, y, N, selected, sums, error, an, bn)
            self.a0_label.configure(text=f'a₀ = {a0:.12g}')
        else:
            self.build_plots(x, y, N, selected, sums, error, a0, an, bn)
        self.table(self.coefficient_tab, ["n", "a_n", "b_n", "|a_n|", "|b_n|"],
                   [[n, a, b, abs(a), abs(b)] for n, (a, b) in enumerate(zip(an, bn), start=1)])
        self.table(self.convergence_tab, ["N", "MSE", "RMSE"], convergence)
        self.last_result = (x,y,sums[N],error,a0,an,bn,metrics,convergence)
        self.last_settings = settings
        self.information.set(f"f(x) = {expression}    구간: [{-L:.6g}, {L:.6g}]    period = 2L = {2*L:.6g}    current N = {N} · 수치 판별: {symmetry(y)}")
        self.advanced_panel.set_data((expression, x, y, L, N, a0, an, bn))
        self.show_advanced()

    def build_plots(self, x, y, N, selected, sums, error, a0, an, bn):
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
        self.a0_label = ttk.Label(self.coefficient_tab, text=f"a₀ = {a0:.12g}", padding=8)
        self.a0_label.pack(anchor="w")
        spectrum = ttk.Frame(self.coefficient_tab, height=280)
        spectrum.pack(fill="both", expand=True)
        self.embed_figure(spectrum, make_spectrum_figure(an, bn))

    def refresh_plots(self, x, y, N, selected, sums, error, an, bn):
        """Figure와 Canvas를 유지하고 선 개수가 같으면 데이터만 갱신한다."""
        for figure, orders, title in zip(self.figures[:2], [[N],sorted(selected)],
                                         [f'Current approximation: N={N}', 'Convergence comparison']):
            ax = figure.axes[0]
            values = [('f(x)',y), *[(f'N = {n}',sums[n]) for n in orders]]
            if len(ax.lines) != len(values):
                for line in list(ax.lines): line.remove()
                for label, data in values: ax.plot(x,data,label=label, color='black' if label=='f(x)' else None)
            else:
                for line,(label,data) in zip(ax.lines,values):
                    line.set_data(x,data); line.set_label(label)
            ax.set_title(title); ax.relim(); ax.autoscale_view(); ax.legend()
        ax = self.figures[2].axes[0]
        ax.lines[0].set_data(x,error)
        ax.relim(); ax.autoscale_view()
        ax = self.figures[3].axes[0]
        ax.clear()
        n = np.arange(1,len(an)+1)
        ax.stem(n,np.abs(an),linefmt='C0-',markerfmt='C0o',basefmt=' ',label='|a_n|')
        ax.stem(n,np.abs(bn),linefmt='C1--',markerfmt='C1x',basefmt=' ',label='|b_n|')
        ax.set(title='Coefficient spectrum',xlabel='n',ylabel='Coefficient magnitude')
        ax.grid(alpha=.3); ax.legend()
        for figure in self.figures:
            figure.canvas.toolbar.update()
            figure.canvas.draw_idle()

    def current_settings(self):
        panel = self.advanced_panel
        return dict(version=1, function=self.expression.get(), L=self.length.get(), N=self.order.get(),
                    num_points=self.points.get(), comparison_N=[n for n,v in self.comparisons.items() if v.get()],
                    advanced={name:getattr(panel,name).get() for name in ['mode','M','center','width','orders','frequency','rate']})

    def save_settings(self):
        try:
            settings = validate_settings(self.current_settings())
            path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('설정 JSON','*.json')])
            if path: Path(path).write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
        except Exception as exc: messagebox.showerror('설정 저장 오류',str(exc))

    def apply_settings(self, settings, asynchronous=False):
        # 전체를 검증한 뒤 적용하여 잘못된 파일로 일부만 변경되는 것을 방지한다.
        validate_settings(settings)
        self.expression.set(settings['function']); self.length.set(settings['L'])
        self.order.set(settings['N']); self.points.set(settings['num_points'])
        for n,value in self.comparisons.items(): value.set(n in settings['comparison_N'])
        for name,value in settings['advanced'].items():
            if name in ['mode','M','center','width','orders','frequency','rate']:
                getattr(self.advanced_panel,name).set(str(value))
        self.example.set('직접 입력')
        self.advanced_panel.update_inputs()
        self.recalculate(asynchronous=asynchronous)

    def load_settings(self):
        path = filedialog.askopenfilename(filetypes=[('설정 JSON','*.json')])
        if not path: return
        try: self.apply_settings(json.loads(Path(path).read_text(encoding='utf-8-sig')),asynchronous=True)
        except Exception as exc: messagebox.showerror('설정 불러오기 오류',str(exc))

    def save_results(self):
        # 편집 중인 입력 대신 마지막으로 계산에 성공한 설정과 결과를 저장한다.
        if self.last_result is None:
            messagebox.showinfo('결과 저장','유효한 함수를 먼저 계산하십시오.'); return
        path = filedialog.asksaveasfilename(defaultextension='.zip', filetypes=[('CSV 결과 묶음','*.zip')])
        if not path: return
        try:
            export_results(path,self.last_settings,self.last_result)
            messagebox.showinfo('저장 완료','CSV 4개와 당시 설정 JSON을 저장했습니다. 마지막으로 성공한 급수 계산 결과입니다.')
        except Exception as exc: messagebox.showerror('결과 저장 오류',str(exc))

    def close(self):
        self.cancel_calculation()
        for window in self.animations:
            if window.winfo_exists(): window.close()
        if self.pending_update is not None:
            self.after_cancel(self.pending_update)
        self.clear_results()
        self.destroy()

    def request_animation(self):
        data = self.advanced_panel.data
        if data is None:
            messagebox.showinfo('애니메이션','유효한 함수를 먼저 계산하십시오.'); return
        self.calculate_job.start('animation',data,
            lambda coefficients:self.open_animation(data,coefficients),
            lambda message:messagebox.showerror('애니메이션 오류',message))

    def open_animation(self, data=None, coefficients=None):
        if self.advanced_panel.data is None:
            messagebox.showinfo('애니메이션','유효한 함수를 먼저 계산하십시오.'); return
        try:
            self.animations = [window for window in self.animations if window.winfo_exists()]
            window = AnimationWindow(self,self.advanced_panel.data if data is None else data,coefficients)
            self.animations.append(window)
            return window
        except Exception as exc: messagebox.showerror('애니메이션 오류',str(exc))

    def save_transform(self):
        snapshot = self.advanced_panel.last_transform
        if snapshot is None:
            messagebox.showinfo('DFT 저장','확장 탭에서 DFT·FFT·급수 비교를 먼저 실행하십시오.'); return
        path = filedialog.asksaveasfilename(defaultextension='.zip',filetypes=[('DFT CSV 묶음','*.zip')])
        if not path: return
        try:
            export_transform(path,snapshot)
            messagebox.showinfo('저장 완료','마지막 성공한 DFT의 변환 표, 표본, 메타데이터를 저장했습니다.')
        except Exception as exc: messagebox.showerror('DFT 저장 오류',str(exc))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    FourierDesktop().mainloop()

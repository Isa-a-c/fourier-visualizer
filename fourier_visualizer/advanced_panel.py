"""확장 실험 UI: 탭이 열릴 때만 계산하여 기본 화면의 반복 작업을 줄인다."""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from advanced import make_analysis_figure
from background import BackgroundJob, calculate_request


class AdvancedPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.data = None
        self.last_transform = None
        self.figure = None
        self.mode = tk.StringVar(value='Gibbs 확대')
        self.M = tk.StringVar(value='64')
        self.frequency = tk.StringVar(value='9')
        self.rate = tk.StringVar(value='12')
        self.center = tk.StringVar(value='0')
        self.width = tk.StringVar(value='0.5')
        self.orders = tk.StringVar(value='1,10,50')
        self.canvas = None
        self.note = tk.StringVar(value='함수를 계산한 뒤 실험을 선택하십시오.')
        self.job = BackgroundJob(self,self.task_state)
        controls = ttk.Frame(self, padding=8)
        controls.pack(fill='x')
        box = ttk.Combobox(controls, state='readonly', textvariable=self.mode, width=24,
                          values=['Gibbs 확대', '진폭·위상', 'DFT·FFT·급수 비교', 'Aliasing 실험', '적분 안정성', 'N별 오차 곡선'])
        box.grid(row=0, column=0, columnspan=2, sticky='w')
        box.bind('<<ComboboxSelected>>', self.request_render)
        self.inputs = {}
        for name, label, var in [
            ('M', 'DFT 표본 수 M (16~256)', self.M), ('center', '확대 중심', self.center),
            ('width', '확대 반폭', self.width), ('orders', '비교 N (쉼표 구분)', self.orders),
            ('frequency', '신호 f (Hz)', self.frequency), ('rate', '샘플링 fs (Hz)', self.rate),
        ]:
            frame = ttk.Frame(controls)
            caption = ttk.Label(frame, text=label)
            caption.pack(side='left', padx=4)
            entry = ttk.Entry(frame, textvariable=var, width=14)
            entry.pack(side='left')
            entry.bind('<Return>',lambda event:(self.request_render(), 'break')[1])
            self.inputs[name] = (frame, caption)
        ttk.Button(controls, text='실험 갱신', command=self.request_render).grid(row=0, column=4)
        ttk.Button(controls, text='취소', command=self.cancel).grid(row=0, column=5)
        self.progress = ttk.Progressbar(self,mode='indeterminate')
        self.progress.pack(fill='x')
        ttk.Label(self, textvariable=self.note, wraplength=820, padding=8).pack(fill='x')
        self.plot = ttk.Frame(self)
        self.plot.pack(fill='both', expand=True)
        self.update_inputs()

    def update_inputs(self):
        for frame, _ in self.inputs.values(): frame.grid_remove()
        visible = {
            'Gibbs 확대': ['center', 'width', 'orders'], '진폭·위상': [],
            'DFT·FFT·급수 비교': ['M'], 'Aliasing 실험': ['M', 'frequency', 'rate'],
            '적분 안정성': [], 'N별 오차 곡선': [],
        }[self.mode.get()]
        self.inputs['M'][1].configure(text='실험 표본 수 M (16~256)' if self.mode.get() == 'Aliasing 실험' else 'DFT 표본 수 M (16~256)')
        for index, name in enumerate(visible):
            self.inputs[name][0].grid(row=1+index//2, column=(index%2)*3, columnspan=3, sticky='w', pady=4)

    def clear(self):
        for widget in self.plot.winfo_children(): widget.destroy()
        if self.figure is not None: plt.close(self.figure)
        self.figure = None
        self.canvas = None

    def set_data(self, data):
        self.cancel(notify=False)
        self.data = data
        self.last_transform = None
        if data is None:
            self.clear()
            self.note.set('유효한 함수를 먼저 입력하십시오.')

    def task_state(self,busy,text):
        self.note.set(text)
        if busy: self.progress.start(12)
        else: self.progress.stop()

    def cancel(self,notify=True):
        self.job.cancel(notify=notify)
        self.progress.stop()

    def request_render(self, *_):
        self.render(asynchronous=True)

    def render(self, *_, asynchronous=False):
        self.cancel(notify=False)
        self.update_inputs()
        if self.data is None: return
        try:
            options = {}
            if self.mode.get() in ['DFT·FFT·급수 비교', 'Aliasing 실험']:
                M = int(self.M.get())
                if not 16 <= M <= 256: raise ValueError('M은 16~256 정수로 입력하십시오.')
                options['M'] = M
            if self.mode.get() == 'Gibbs 확대':
                options.update(center=float(self.center.get()), width=float(self.width.get()))
                options['orders'] = [int(value.strip()) for value in self.orders.get().split(',')]
            if self.mode.get() == 'Aliasing 실험':
                options.update(frequency=float(self.frequency.get()), sampling_rate=float(self.rate.get()))
            mode,data = self.mode.get(),self.data
            if asynchronous:
                self.job.start('analysis',(mode,data,options),
                    lambda prepared:self.display_result(mode,data,options,prepared),self.display_error)
            else:
                self.display_result(mode,data,options,calculate_request('analysis',(mode,data,options)))
        except Exception as exc:
            self.display_error(str(exc))

    def display_result(self,mode,data,options,prepared):
        try:
            self.figure, note = make_analysis_figure(mode, data, figure=self.figure, prepared=prepared, **options)
            if mode == 'DFT·FFT·급수 비교':
                self.last_transform = (data[0], data[3], options['M'])
            self.note.set(note)
            if self.canvas is None:
                self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot)
                self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot, pack_toolbar=False)
                self.toolbar.pack(side='bottom', fill='x')
                self.canvas.get_tk_widget().pack(fill='both', expand=True)
            self.toolbar.update()
            self.canvas.draw_idle()
        except Exception as exc:
            self.display_error(str(exc))

    def display_error(self,message):
        if self.mode.get() == 'DFT·FFT·급수 비교': self.last_transform = None
        self.clear()
        self.note.set(f'실험 입력을 확인하십시오: {message}')

"""고정된 입력의 Fourier 부분합을 한 항씩 더하는 별도 창."""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from fourier_core import calculate_fourier_coefficients


class AnimationWindow(tk.Toplevel):
    def __init__(self, parent, data, coefficients=None):
        super().__init__(parent)
        self.withdraw()
        self.job = None
        self.playing = False
        self.figure = None
        self.protocol('WM_DELETE_WINDOW', self.close)
        try:
            expression, self.x, y, L, _, _, _, _ = data
            self.a0, self.an, self.bn = coefficients if coefficients is not None else calculate_fourier_coefficients(self.x,y,L,100)
            self.angle = np.pi*self.x/L
            self.N = 0
            self.values = np.full_like(self.x,self.a0)
            self.delay = tk.IntVar(value=200)
            self.auto_y = tk.BooleanVar(value=True)
            self.range_note = tk.StringVar(value='자동 세로축: 원래 함수와 현재 부분합을 함께 표시합니다.')
            self.original_y = y
            self.title(f'부분합 애니메이션 · f(x)={expression}')
            self.geometry('950x620')
            controls = ttk.Frame(self,padding=8)
            controls.pack(fill='x')
            ttk.Button(controls,text='재생',command=self.play).pack(side='left')
            ttk.Button(controls,text='정지',command=self.pause).pack(side='left')
            ttk.Button(controls,text='처음으로',command=self.reset).pack(side='left')
            ttk.Label(controls,text='프레임 간격(ms)').pack(side='left',padx=10)
            tk.Scale(controls,from_=50,to=1000,resolution=50,orient='horizontal',variable=self.delay).pack(side='left')
            ttk.Checkbutton(controls,text='세로축 자동 확장',variable=self.auto_y,command=self.update_limits).pack(side='left')
            ttk.Label(self,textvariable=self.range_note).pack()
            ttk.Label(self,text='창을 열었을 때의 함수·구간·표본을 고정하여 N=0→100을 재생합니다. 메인 창 설정은 변경되지 않습니다.').pack()
            self.figure, self.axes = plt.subplots(figsize=(9,5))
            self.axes.plot(self.x,y,'k',label='f(x)')
            self.line, = self.axes.plot(self.x,self.values,label='S_N(x)')
            margin = max(float(np.ptp(y))*.2,.2)
            self.axes.set_ylim(float(np.min(y))-margin,float(np.max(y))+margin)
            self.initial_limits = self.axes.get_ylim()
            self.axes.set(xlabel='x',ylabel='Function value',title='N = 0 (a0)')
            self.axes.legend(); self.axes.grid(alpha=.3)
            self.figure.tight_layout()
            self.update_limits()
            self.canvas = FigureCanvasTkAgg(self.figure,master=self)
            toolbar = NavigationToolbar2Tk(self.canvas,self,pack_toolbar=False)
            toolbar.pack(side='bottom',fill='x')
            self.canvas.get_tk_widget().pack(fill='both',expand=True)
            self.canvas.draw()
            self.deiconify()
        except Exception:
            self.close()
            raise

    def step(self):
        self.job = None
        if not self.playing: return
        if self.N >= 100:
            self.pause(); return
        self.N += 1
        n = self.N
        self.values += self.an[n-1]*np.cos(n*self.angle)+self.bn[n-1]*np.sin(n*self.angle)
        self.line.set_ydata(self.values)
        self.axes.set_title(f'N = {n}')
        self.update_limits()
        self.canvas.draw_idle()
        if n == 100: self.pause()
        else: self.job = self.after(self.delay.get(),self.step)

    def play(self):
        if self.playing: return
        if self.N >= 100: self.reset()
        self.playing = True
        self.step()

    def pause(self):
        self.playing = False
        if self.job is not None:
            self.after_cancel(self.job)
            self.job = None

    def reset(self):
        self.pause()
        self.N = 0
        self.values.fill(self.a0)
        self.line.set_ydata(self.values)
        self.axes.set_title('N = 0 (a0)')
        self.axes.set_ylim(*self.initial_limits)
        self.update_limits()
        self.canvas.draw_idle()

    def update_limits(self):
        low = min(float(np.min(self.original_y)),float(np.min(self.values)))
        high = max(float(np.max(self.original_y)),float(np.max(self.values)))
        bottom,top = self.axes.get_ylim()
        clipped = low < bottom or high > top
        if self.auto_y.get() and clipped:
            margin=max((high-low)*.08,.05)
            self.axes.set_ylim(min(bottom,low-margin),max(top,high+margin))
            self.range_note.set('세로축을 확장했습니다. 원래 함수와 현재 부분합이 모두 표시됩니다.')
        elif not self.auto_y.get() and clipped:
            self.range_note.set('주의: 현재 부분합 또는 원래 함수가 세로축 범위 밖에 있습니다. 자동 확장을 켜시면 모두 표시됩니다.')
        else:
            self.range_note.set('원래 함수와 현재 부분합이 세로축 범위 안에 있습니다.')
        if hasattr(self,'canvas'): self.canvas.draw_idle()

    def close(self):
        self.pause()
        if self.figure is not None: plt.close(self.figure)
        self.destroy()

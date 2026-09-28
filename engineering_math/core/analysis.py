"""v0.4: 급수 분석과 DFT/FFT/aliasing 실험. 주파수 단위는 cycles/x."""
import numpy as np
from engineering_math.core.fourier import sample_function, calculate_fourier_sums, calculate_fourier_coefficients
from engineering_math.core.diagnostics import integration_stability, convergence_errors


def symmetry(y, tolerance=1e-7):
    """대칭 격자의 표본으로 판별한다. 수학적 증명이 아닌 수치 추정이다."""
    scale = max(1.0, float(np.max(np.abs(y))))
    even = np.max(np.abs(y - y[::-1])) <= tolerance * scale
    odd = np.max(np.abs(y + y[::-1])) <= tolerance * scale
    return '우함수·기함수 모두 (영함수 수준)' if even and odd else '우함수' if even else '기함수' if odd else '둘 다 아님'


def amplitude_phase(an, bn):
    """a cos θ + b sin θ = A cos(θ-φ). 작은 진폭의 위상은 정의하지 않는다."""
    amplitude = np.hypot(an, bn)
    phase = np.arctan2(bn, an)
    phase[amplitude <= 1e-10 * max(1.0, float(np.max(amplitude)))] = np.nan
    return amplitude, phase


def transform(expression, L, M):
    """같은 M개 표본의 직접 DFT와 FFT. 중복 주기 끝점은 제외한다."""
    if not isinstance(M, (int, np.integer)) or not 16 <= M <= 256:
        raise ValueError('직접 DFT 비교의 M은 16~256 정수여야 합니다.')
    if not np.isfinite(L) or L <= 0:
        raise ValueError('L은 유한한 양수여야 합니다.')
    x = np.linspace(-L, L, M, endpoint=False)
    y = sample_function(expression, x)
    indices = np.arange(M)
    with np.errstate(over='raise', invalid='raise'):
        direct = np.exp(-2j * np.pi * np.outer(indices, indices) / M) @ y
        fast = np.fft.fft(y)
        # 시작 좌표 -L에 따른 위상을 제거하여 급수의 x 기준과 일치시킨다.
        k = np.arange((M - 1) // 2 + 1)
        coefficients = fast[k] / M * np.exp(1j * np.pi * k)
    if not np.all(np.isfinite(direct)) or not np.all(np.isfinite(fast)):
        raise ValueError('변환 결과가 유한하지 않습니다. 함수의 크기를 줄이십시오.')
    return x, y, direct, fast, coefficients


def alias_signal(frequency, sampling_rate, M=64):
    if not np.isfinite(frequency) or frequency < 0 or not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError('주파수는 0 이상, 샘플링 주파수는 양수여야 합니다.')
    # 코사인 신호에서는 ±주파수가 동일하다. Nyquist 구간으로 접힌 양의 주파수.
    alias = abs((frequency + sampling_rate / 2) % sampling_rate - sampling_rate / 2)
    t = np.arange(M) / sampling_rate
    return t, np.cos(2 * np.pi * frequency * t), alias


def prepare_analysis(mode, data, M=64, frequency=9.0, sampling_rate=12.0, center=0.0, width=0.5, orders=None):
    """수치 데이터만 준비한다. 백그라운드에서 호출할 수 있다."""
    expression,x,y,L,N,a0,an,bn = data
    if mode == 'Gibbs 확대':
        if not np.isfinite(center) or not np.isfinite(width) or width <= 0 or not -L <= center <= L:
            raise ValueError('확대 중심은 [-L,L], 반폭은 유한한 양수여야 합니다.')
        orders = sorted(set([1,min(10,N),N] if orders is None else orders))
        if not orders or any(type(n) is not int or not 1 <= n <= 100 for n in orders):
            raise ValueError('비교 N은 1~100 정수를 하나 이상 입력하십시오.')
        if max(orders)>len(an): a0,an,bn=calculate_fourier_coefficients(x,y,L,max(orders))
        grid=np.linspace(max(-L,center-width),min(L,center+width),4000)
        return dict(grid=grid,original=sample_function(expression,grid),
                    sums=calculate_fourier_sums(grid,L,a0,an,bn,orders))
    if mode == '진폭·위상': return dict(phase=amplitude_phase(an[:N],bn[:N]))
    if mode == 'DFT·FFT·급수 비교': return dict(transform=transform(expression,L,M))
    if mode == '적분 안정성': return dict(stability=integration_stability(expression,L,len(x),N))
    if mode == 'N별 오차 곡선': return dict(convergence=convergence_errors(x,y,L,a0,an[:N],bn[:N]))
    if mode == 'Aliasing 실험': return dict(alias=alias_signal(frequency,sampling_rate,M))
    raise ValueError('지원하지 않는 실험입니다.')


def make_analysis_figure(mode, data, M=64, frequency=9.0, sampling_rate=12.0, center=0.0, width=0.5,
                         orders=None, figure=None, prepared=None):
    import matplotlib.pyplot as plt
    expression, x, y, L, N, a0, an, bn = data
    if prepared is None:
        prepared = prepare_analysis(mode,data,M,frequency,sampling_rate,center,width,orders)
    def subplots(rows=1, **kwargs):
        if figure is None:
            return plt.subplots(rows, 1, figsize=(9, 5), **kwargs)
        figure.clear()
        return figure, figure.subplots(rows, 1, **kwargs)
    if mode == 'Gibbs 확대':
        if not np.isfinite(center) or not np.isfinite(width) or width <= 0 or not -L <= center <= L:
            raise ValueError('확대 중심은 [-L,L], 반폭은 유한한 양수여야 합니다.')
        grid, original = prepared['grid'], prepared['original']
        fig, ax = subplots()
        ax.plot(grid, original, 'k', label='f(x)')
        for order, values in prepared['sums'].items():
            ax.plot(grid, values, label=f'N={order}')
        ax.set(xlabel='x', ylabel='Function value', title='Local approximation (adjust center and half-width)')
        ax.grid(alpha=.3); ax.legend()
        note = '확대 관찰용 그래프입니다. 모든 함수에 불연속이 있는 것은 아닙니다. sign(x), 중심 0으로 실험해 보십시오.'
    elif mode == '진폭·위상':
        magnitude, phase = prepared['phase']
        orders = np.arange(1, N+1)
        fig, axes = subplots(2, sharex=True)
        axes[0].stem(orders, magnitude)
        axes[1].plot(orders, phase, 'o')
        axes[0].set(ylabel='Amplitude', title='A cos(theta - phi)')
        axes[1].set(xlabel='Harmonic n', ylabel='Phase (rad)')
        for ax in axes: ax.grid(alpha=.3)
        note = f'표본 대칭 판별: {symmetry(y)} · 위상 φ=atan2(b,a). 진폭이 거의 0인 항의 위상은 생략합니다.'
    elif mode == 'DFT·FFT·급수 비교':
        grid, values, direct, fast, coefficients = prepared['transform']
        count = min(N, len(coefficients)-1)
        k = np.arange(1, count+1)
        dft_an = 2*coefficients[1:count+1].real
        dft_bn = -2*coefficients[1:count+1].imag
        fig, axes = subplots(2)
        bins = np.fft.fftshift(np.fft.fftfreq(M, d=2*L/M))
        axes[0].plot(bins, np.fft.fftshift(np.abs(direct)/M), 'o', label='Direct DFT / M')
        axes[0].plot(bins, np.fft.fftshift(np.abs(fast)/M), '-', label='FFT / M')
        axes[0].set(xlabel='Frequency (cycles / x)', ylabel='Two-sided magnitude')
        axes[1].plot(k, an[:count], 'o-', label='Series a_n')
        axes[1].plot(k, dft_an, 'x--', label='DFT a_n')
        axes[1].plot(k, bn[:count], 'o-', label='Series b_n')
        axes[1].plot(k, dft_bn, 'x--', label='DFT b_n')
        axes[1].set(xlabel='Harmonic n (below Nyquist)', ylabel='Coefficient')
        for ax in axes: ax.legend(); ax.grid(alpha=.3)
        note = (f'M={M}, Δx={2*L/M:.5g}, fs={M/(2*L):.5g} cycles/x, Δf={1/(2*L):.5g}, '
                f'최대 |DFT−FFT|={np.max(np.abs(direct-fast)):.3g}\n'
                f'DC: 급수={a0:.6g}, DFT={coefficients[0].real:.6g} · 비교 n≤{count}. '
                '급수는 끝점 포함 사다리꼴 적분, DFT는 끝점 제외 M개 표본이므로 계수 차이가 발생할 수 있습니다.')
    elif mode == '적분 안정성':
        coefficients, differences, relative = prepared['stability']
        (coarse_a0,coarse_an,coarse_bn), (fine_a0,fine_an,fine_bn) = coefficients
        fig, axes = subplots(2)
        orders = np.arange(1,N+1)
        axes[0].plot(orders,np.abs(fine_an-coarse_an),'o-',label='|delta a_n|')
        axes[0].plot(orders,np.abs(fine_bn-coarse_bn),'x-',label='|delta b_n|')
        axes[0].set(xlabel='Harmonic n',ylabel='Absolute coefficient change')
        axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].bar(['a0','a_n max','b_n max'], [abs(fine_a0-coarse_a0),np.max(np.abs(fine_an-coarse_an)),np.max(np.abs(fine_bn-coarse_bn))])
        axes[1].set(ylabel='Absolute change')
        ratio = f'{relative:.3g}' if np.isfinite(relative) else '정의 생략 (계수≈0)'
        note = (f'표본 {len(x)} → {2*len(x)-1}, N={N} · 최대 계수 변화={np.max(differences):.3g}, '
                f'전체 계수 최대 크기로 정규화한 변화={ratio}. 두 격자 비교만으로 정확성을 보장하지는 않습니다. 같은 alias가 두 격자에 남을 수도 있습니다.')
    elif mode == 'N별 오차 곡선':
        rows = prepared['convergence']
        fig, axes = subplots(2,sharex=True)
        axes[0].plot(rows[:,0],rows[:,1],'-o',markersize=3)
        axes[1].plot(rows[:,0],rows[:,2],'-o',markersize=3)
        axes[0].set(ylabel='MSE')
        axes[1].set(xlabel='N',ylabel='RMSE')
        for ax in axes: ax.grid(alpha=.3)
        note = f'N=1~{N}, 고정된 {len(x)}개 평가 표본의 오차입니다. 불연속점·주기 경계 및 수치 오차 때문에 항상 단조 감소하지는 않을 수 있습니다.'
    elif mode == 'Aliasing 실험':
        t, samples, alias = prepared['alias']
        dense = np.linspace(0, t[-1], 6000)
        fig, ax = subplots()
        ax.plot(dense, np.cos(2*np.pi*frequency*dense), alpha=.65, label=f'Original {frequency:g}')
        ax.plot(dense, np.cos(2*np.pi*alias*dense), '--', label=f'Alias {alias:g}')
        ax.plot(t, samples, 'ko', markersize=3, label='Samples')
        ax.set(xlabel='Time (s)', ylabel='cos(2 pi f t)', title='Aliasing experiment (independent cosine signal)')
        ax.legend(); ax.grid(alpha=.3)
        note = f'독립 코사인 실험: f={frequency:g} Hz, fs={sampling_rate:g} Hz, Nyquist={sampling_rate/2:g} Hz, alias={alias:g} Hz. 위의 사용자 함수와 별개입니다.'
    else:
        raise ValueError('지원하지 않는 실험입니다.')
    fig.tight_layout()
    return fig, note

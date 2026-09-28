"""v0.4 수학과 숨겨진 데스크톱 창 회귀 검사."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'fourier_visualizer'))
import numpy as np
from fourier_core import parse_expression, FourierSession
from advanced import symmetry, amplitude_phase, transform, alias_signal

assert symmetry(np.array([-1.,0.,1.])) == '기함수'
assert symmetry(np.array([1.,0.,1.])) == '우함수'
assert '모두' in symmetry(np.zeros(3))
A, phase = amplitude_phase(np.array([3.,0.]), np.array([4.,0.]))
assert np.allclose(A,[5.,0.]) and np.isnan(phase[1])
for M in [16, 63, 64, 256]:
    for expression, k, value in [('3',0,3), ('cos(3*x)',3,.5), ('sin(2*x)',2,-.5j)]:
        _, _, direct, fast, coefficients = transform(parse_expression(expression), np.pi, M)
        assert np.allclose(direct, fast, atol=1e-10)
        assert np.isclose(coefficients[k],value,atol=1e-12)
for f,fs in [(9,12),(3,12),(6,12),(0,12),(25,12)]:
    t,y,alias=alias_signal(f,fs)
    assert np.allclose(y,np.cos(2*np.pi*alias*t),atol=1e-12)
from desktop_app import FourierDesktop
window=FourierDesktop()
window.withdraw()
window.update()
window.expression.set('sign(x)')
window.recalculate()
panel=window.advanced_panel
for mode in ['Gibbs 확대','진폭·위상','DFT·FFT·급수 비교','Aliasing 실험']:
    panel.mode.set(mode)
    panel.render()
    assert panel.figure is not None, panel.note.get()
panel.M.set('0')
panel.render()
assert panel.figure is None and '입력' in panel.note.get()
window.reset_defaults()
assert panel.M.get()=='64'
window.expression.set('sqrt(x)')
window.recalculate()
assert panel.data is None and panel.figure is None
window.close()
print('PASS: symmetry, phase, DFT=FFT, origin correction, alias equality, all GUI modes and errors')

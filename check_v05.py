import sys
import csv
import io
import json
import zipfile
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'fourier_visualizer'))
import numpy as np
from diagnostics import integration_stability, convergence_errors
from fourier_core import parse_expression, calculate_fourier_coefficients, calculate_fourier_sum, calculate_errors
from project_io import export_transform

expression=parse_expression('x**2')
coefficients,difference,relative=integration_stability(expression,np.pi,500,20)
assert coefficients[1][0] > np.pi**2/3
assert abs(coefficients[1][0]-np.pi**2/3) < abs(coefficients[0][0]-np.pi**2/3)
assert np.isfinite(relative) and difference.shape==(41,)
x=np.linspace(-np.pi,np.pi,2000)
a0,an,bn=calculate_fourier_coefficients(x,x,np.pi,30)
rows=convergence_errors(x,x,np.pi,a0,an,bn)
assert rows.shape==(30,3)
for n in [1,10,30]:
    _,metrics=calculate_errors(x,calculate_fourier_sum(x,np.pi,a0,an,bn,n))
    assert np.allclose(rows[n-1,1:],[metrics['MSE'],metrics['RMSE']])
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
    path=Path(directory)/'dft.zip'
    export_transform(path,(parse_expression('sin(2*x)'),np.pi,64))
    with zipfile.ZipFile(path) as archive:
        entries=list(csv.DictReader(io.StringIO(archive.read('transform.csv').decode('utf-8-sig'))))
        assert len(entries)==64
        assert np.isclose(float(entries[2]['c_x_imag']),-.5)
        assert np.isclose(float(entries[2]['phase_rad']),-np.pi/2)
        assert entries[0]['phase_rad']==''
        assert json.loads(archive.read('metadata.json'))['M']==64

from desktop_app import FourierDesktop
window=FourierDesktop()
window.withdraw()
window.update()
window.recalculate()
for mode in ['적분 안정성','N별 오차 곡선']:
    window.advanced_panel.mode.set(mode)
    window.advanced_panel.render()
    assert window.advanced_panel.figure is not None,window.advanced_panel.note.get()
window.advanced_panel.mode.set('DFT·FFT·급수 비교')
window.advanced_panel.render()
snapshot=window.advanced_panel.last_transform
window.advanced_panel.M.set('128')
assert window.advanced_panel.last_transform==snapshot
animation=window.open_animation()
animation.withdraw()
animation.play()
animation.pause()
assert animation.N==1 and animation.job is None
expected=calculate_fourier_sum(animation.x,np.pi,animation.a0,animation.an,animation.bn,1)
assert np.allclose(animation.values,expected)
animation.reset()
assert animation.N==0 and np.allclose(animation.values,animation.a0)
for _ in range(100):
    animation.play()
    animation.pause()
assert animation.N==100
animation.play(); animation.pause()
assert animation.N==1
window.close()
assert not animation.playing and animation.job is None
print('PASS: nested-grid stability, convergence, DFT CSV/phase, snapshot, animation step/pause/reset/end/close')

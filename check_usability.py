import csv
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'fourier_visualizer'))
from desktop_app import FourierDesktop
from project_io import export_results

window=FourierDesktop()
window.withdraw()
window.update()
window.recalculate()
figures=list(window.figures)
trees=dict(window.tables)
window.order.set(30)
window.recalculate()
assert all(a is b for a,b in zip(figures,window.figures))
assert trees==window.tables
assert len(window.figures[0].axes[0].lines[1].get_ydata())==5000
panel=window.advanced_panel
panel.orders.set('2,40,100')
panel.render()
assert panel.figure is not None,panel.note.get()
assert len(panel.figure.axes[0].lines)==4
figure=panel.figure
canvas=panel.canvas
panel.width.set('0.2')
panel.render()
assert panel.figure is figure and panel.canvas is canvas
for mode,visible in [('진폭·위상',set()),('DFT·FFT·급수 비교',{'M'}),('Aliasing 실험',{'M','frequency','rate'})]:
    panel.mode.set(mode)
    panel.render()
    assert {name for name,(frame,_) in panel.inputs.items() if frame.grid_info()}==visible
    assert panel.figure is not None,panel.note.get()
settings=window.current_settings()
window.reset_defaults()
window.apply_settings(json.loads(json.dumps(settings)))
assert window.current_settings()==settings
bad=dict(settings,N=200)
try: window.apply_settings(bad)
except ValueError: pass
else: raise AssertionError('Invalid settings accepted')
assert window.current_settings()==settings
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
    path=Path(directory)/'results.zip'
    export_results(path,window.last_settings,window.last_result)
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist())==5
        assert json.loads(archive.read('settings.json'))==settings
        rows=list(csv.DictReader(io.StringIO(archive.read('samples.csv').decode('utf-8-sig'))))
        assert len(rows)==settings['num_points']
        for row in rows[:10]:
            assert abs(float(row['f(x)'])-float(row['S_N(x)'])-float(row['error'])) < 1e-12
window.close()
print('PASS: figure/canvas/table reuse, mode-specific inputs, Gibbs orders, JSON round trip, invalid settings, CSV export')

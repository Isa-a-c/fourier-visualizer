from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'fourier_visualizer'))
from desktop_app import FourierDesktop

window = FourierDesktop()
window.withdraw()
window.update()
window.recalculate()
assert not window.status.get(), window.status.get()
assert len(window.figures) == 4
window.order_input.delete(0, 'end')
window.order_input.insert(0, '37')
assert window.order.get() == 37
window.recalculate()
assert not window.status.get()
assert 'current N = 37' in window.information.get()
window.order.set(12)
assert window.order_text.get() == '12'
for text in ['0', '101', '-1', '1.5', 'abc']:
    assert not window.validate_order_input(text)
window.order_input.delete(0, 'end')
window.finish_order_input()
assert window.order_text.get() == '12'
window.order.set(100)
for variable in window.comparisons.values():
    variable.set(False)
window.recalculate()
assert not window.status.get()
for expression in ['3', 'sign(x)', 'sin(x)', 'x**2']:
    window.expression.set(expression)
    window.recalculate()
    assert not window.status.get(), window.status.get()
window.expression.set('sqrt(x)')
window.recalculate()
assert window.status.get() and not window.figures
window.expression.set('x')
window.recalculate()
assert not window.status.get() and len(window.figures) == 4
window.close()
print('Desktop GUI: initial render, N=100, empty comparisons, functions, errors and recovery PASS')

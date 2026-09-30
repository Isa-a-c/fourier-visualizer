"""확장 함수·매개변수·사용자 외력과 부분 성공 상태의 회귀 검증입니다."""
import numpy as np
import sympy as sp
import pytest
from scipy import special
from engineering_math.core.expr import parse_expression, parameter_expression, parse_parameters, evaluate
from engineering_math.topics.fourier import FourierTopic
from engineering_math.topics.heat import HeatTopic
from engineering_math.topics.wave import WaveTopic
from engineering_math.topics.ode import OdeTopic, FORCES
from engineering_math.topics.laplace import LaplaceTopic


@pytest.mark.parametrize('expression, expected', [
    ('asin(x)', np.arcsin), ('arccos(x)', np.arccos), ('atan(x)', np.arctan),
    ('asinh(x)', np.arcsinh), ('acosh(x+1)', lambda x: np.arccosh(x+1)),
    ('atanh(x)', np.arctanh), ('ln(x+1)', np.log1p),
    ('sinc(x)', lambda x: np.sinc(x/np.pi)), ('floor(x)', np.floor), ('ceil(x)', np.ceil),
    ('erfc(x)', special.erfc), ('erfi(x)', special.erfi),
    ('besseli(0,x)', lambda x: special.iv(0,x)), ('besselk(0,x+1)', lambda x: special.kv(0,x+1)),
    ('loggamma(x+1)', lambda x: special.loggamma(x+1)),
    ('beta(x+1,2)', lambda x: special.beta(x+1,2)),
])
def test_extended_functions(expression, expected):
    x = np.array([0., .2, .5])
    np.testing.assert_allclose(evaluate(parse_expression(expression), (x,)), expected(x), atol=1e-13)


def test_parameters_power_and_piecewise():
    expression = parameter_expression('A*Piecewise((x^2, x<0), (sin(w*x), True))', 'A=2; w=pi')
    np.testing.assert_allclose(evaluate(expression, (np.array([-1., 0., .5]),)), [2, 0, 2])


@pytest.mark.parametrize('text', ['x=2', 'sin=3', 'A=1; A=2', 'A=1; B=A', 'A=I', 'A=1/0', 'A=2,B=3', '__x=2'])
def test_invalid_parameter_declarations(text):
    with pytest.raises((ValueError, SyntaxError)):
        parse_parameters(text)


def test_parameter_change_invalidates_fourier_cache():
    topic = FourierTopic()
    first = topic.compute({'function': 'A*sin(x)', 'parameters': 'A=2'})
    second = topic.compute({'function': 'A*sin(x)', 'parameters': 'A=3'})
    assert first.data['bn'][0] == pytest.approx(2.)
    assert second.data['bn'][0] == pytest.approx(3.)
    assert second.data['cache_reused'] == 0


@pytest.mark.parametrize('topic', [HeatTopic(), WaveTopic()])
def test_pde_parameters(topic):
    result = topic.compute({'initial': 'A*sin(pi*x/ell)', 'parameters': 'A=2'})
    np.testing.assert_allclose(result.data['initial'], 2*np.sin(np.pi*result.data['x']), atol=1e-14)


def test_laplace_parameters_and_distribution():
    result = LaplaceTopic().compute({'expression': 'A*exp(-w*t)', 'parameters': 'A=2; w=3'})
    s = sp.Symbol('s')
    assert sp.simplify(result.data['transformed']-2/(s+3)) == 0
    result = LaplaceTopic().compute({'expression': 'DiracDelta(t)'})
    assert result.data['transformed'] == 1
    assert 'values' not in result.data


def custom(**params):
    return OdeTopic().compute({'mode': FORCES[4], 'end_time': 2., 'num_points': 201, **params})


def test_manufactured_custom_force():
    result = custom(force_expression='A*sin(t)+b*cos(t)', parameters='A=3; b=2/5', position=0., velocity=1.)
    t = result.data['time']
    np.testing.assert_allclose(result.data['values'][0], np.sin(t), atol=1e-8)
    np.testing.assert_allclose(result.data['values'][1], np.cos(t), atol=1e-8)
    assert result.data['laplace_input'] is None


def test_narrow_piecewise_pulse_auto_boundaries():
    result = custom(force_expression='Piecewise((1, And(t>=tau,t<tau+0.001)), (0, True))',
                    parameters='tau=1/3', damping=0., position=0.)
    t = result.data['time']
    def step_response(delay):
        return np.where(t >= delay, (1-np.cos(2*(t-delay)))/4, 0.)
    exact = step_response(1/3)-step_response(1/3+.001)
    np.testing.assert_allclose(result.data['values'][0], exact, atol=1e-9)
    assert len(result.data['boundaries']) == 4


def test_manual_breakpoints_and_symbolic_success():
    result = custom(force_expression='Heaviside(t-tau)', parameters='tau=1', breakpoints='tau; 1.5',
                    symbolic_mode='라플라스 변환도 계산', position=0.)
    assert result.data['boundaries'] == [0., 1., 1.5, 2.]
    assert result.data['laplace_input'] is not None


def test_symbolic_failure_preserves_numeric_result(monkeypatch):
    def fail(*args, **kwargs):
        raise NotImplementedError('test')
    monkeypatch.setattr(sp, 'laplace_transform', fail)
    result = custom(force_expression='sin(t)', symbolic_mode='라플라스 변환도 계산')
    assert np.all(np.isfinite(result.data['values']))
    assert result.data['laplace_input'] is None
    result = LaplaceTopic().compute({'expression': 'sin(t)'})
    assert 'values' in result.data
    assert result.data['transformed'].has(sp.LaplaceTransform)


@pytest.mark.parametrize('expression', ['1/t', 'sqrt(-t-1)', 'DiracDelta(t)', 'y+t'])
def test_invalid_custom_force(expression):
    with pytest.raises(ValueError):
        custom(force_expression=expression)


def test_parameter_settings_roundtrip(tmp_path):
    from engineering_math.core.project import save_settings, load_settings, export_results
    result = custom(force_expression='A*cos(w*t)', parameters='A=2; w=3')
    path = tmp_path/'parameters.json'
    save_settings(path, 'ode', result.params)
    assert load_settings(path)['params']['parameters'] == 'A=2; w=3'
    export_results(tmp_path/'parameters.zip', result)

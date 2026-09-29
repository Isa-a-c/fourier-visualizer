import numpy as np
import sympy as sp
import pytest
from engineering_math.topics.ode import OdeTopic, FORCES
from engineering_math.topics.laplace import LaplaceTopic, MODES


def test_undamped_oscillator():
    result = OdeTopic().compute({'damping': 0.})
    t = result.data['time']
    np.testing.assert_allclose(result.data['values'][0], np.cos(2*t), atol=1e-8)
    np.testing.assert_allclose(result.data['energy'], 2., atol=1e-8)
    assert result.data['regime'] == '무감쇠'


@pytest.mark.parametrize('damping,regime', [(.4, '부족감쇠'), (4., '임계감쇠'), (6., '과감쇠')])
def test_damping_regimes(damping, regime):
    result = OdeTopic().compute({'damping': damping})
    assert result.data['regime'] == regime
    assert np.all(np.diff(result.data['energy']) < 1e-10)
    assert result.metrics['변위 최대 비교 오차'] < 1e-7
    if damping == 4:
        t = result.data['time']
        np.testing.assert_allclose(result.data['values'][0], (1+2*t)*np.exp(-2*t), atol=1e-8)


def test_resonance():
    result = OdeTopic().compute({'damping': 0., 'position': 0., 'mode': FORCES[2]})
    t = result.data['time']
    exact = np.sin(2*t)/8-t*np.cos(2*t)/4
    np.testing.assert_allclose(result.data['values'][0], exact, atol=1e-8)


@pytest.mark.parametrize('delay', [0., 1.234, 10., 20.])
def test_step_event(delay):
    result = OdeTopic().compute({'damping': 0., 'position': 0., 'mode': FORCES[3], 'delay': delay})
    t = result.data['time']
    exact = np.where(t >= delay, (1-np.cos(2*(t-delay)))/4, 0.)
    np.testing.assert_allclose(result.data['values'][0], exact, atol=1e-8)
    np.testing.assert_allclose(result.data['reference'][0], exact, atol=1e-12)


def test_initial_conditions_laplace_link():
    result = OdeTopic().compute({'mass': 2., 'damping': 0., 'stiffness': 8., 'position': 3., 'velocity': 4.})
    inverse = LaplaceTopic().compute({'mode': MODES[1], 'expression': result.data['laplace_input']})
    t = inverse.data['time']
    np.testing.assert_allclose(inverse.data['values'], 3*np.cos(2*t)+2*np.sin(2*t), atol=1e-12)


@pytest.mark.parametrize('expression,expected,plane', [('1', '1/s', 0), ('exp(-2*t)', '1/(s+2)', -2), ('sin(2*t)', '2/(s**2+4)', 0)])
def test_forward_transforms(expression, expected, plane):
    result = LaplaceTopic().compute({'expression': expression})
    s = sp.Symbol('s')
    assert sp.simplify(result.data['transformed']-sp.sympify(expected, locals={'s': s})) == 0
    assert result.tables['기호 변환'][1][2][1] == f'Re(s) > {plane}'


def test_delayed_inverse():
    result = LaplaceTopic().compute({'mode': MODES[1], 'expression': 'exp(-2*s)/s'})
    t = result.data['time']
    np.testing.assert_allclose(result.data['values'], np.heaviside(t-2, .5))


def test_distribution_not_plotted():
    result = LaplaceTopic().compute({'mode': MODES[1], 'expression': '1'})
    assert result.data['time_expression'].has(sp.DiracDelta)
    assert 'values' not in result.data


@pytest.mark.parametrize('params', [{'mass': 0}, {'damping': -1}, {'stiffness': 0}])
def test_invalid_ode(params):
    with pytest.raises(ValueError):
        OdeTopic().compute(params)


@pytest.mark.parametrize('topic', [OdeTopic(), LaplaceTopic()])
def test_new_topic_settings_and_export(topic, tmp_path):
    import json
    import zipfile
    from engineering_math.core.project import save_settings, load_settings, export_results
    result = topic.compute({})
    path = tmp_path/'설정.json'
    save_settings(path, result.topic, result.params)
    assert load_settings(path)['params'] == result.params
    archive_path = tmp_path/'결과.zip'
    export_results(archive_path, result)
    with zipfile.ZipFile(archive_path) as archive:
        assert json.loads(archive.read('settings.json'))['topic'] == topic.id
        assert json.loads(archive.read('metrics.json'))


def test_unevaluated_forward_has_no_claimed_convergence_region():
    result = LaplaceTopic().compute({'expression': 'exp(t**2)', 'end_time': 1.})
    assert result.data['transformed'].has(sp.LaplaceTransform)
    assert '미확정' in result.tables['기호 변환'][1][2][1]

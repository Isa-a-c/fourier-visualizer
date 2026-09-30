"""PDE 해석해·초기조건·경계조건·에너지와 화면·저장 계약 검증."""
import numpy as np
import pytest
from scipy.special import jn_zeros, jv, erfc
from engineering_math.topics import get_topic
from engineering_math.core.chapter12 import (dalembert, infinite_heat, rectangle_membrane,
    disk_membrane, disk_potential, sphere_potential)


@pytest.mark.parametrize('speed', [.3, 2.])
def test_dalembert_right_travelling_wave(speed):
    x, time = np.linspace(-2, 2, 101), np.linspace(0, 1, 21)
    frames, _ = dalembert(np.sin, lambda x: -speed*np.cos(x), x, time, speed, 32001)
    np.testing.assert_allclose(frames, np.sin(x[None,:]-speed*time[:,None]), atol=2e-8)
    np.testing.assert_allclose(frames[0], np.sin(x), atol=1e-14)


def test_constant_initial_velocity():
    x, time = np.linspace(-2, 2, 21), np.linspace(0, 1, 11)
    frames, _ = dalembert(lambda x: np.full_like(x, 3.), lambda x: np.full_like(x, 2.), x, time, .7, 1001)
    np.testing.assert_allclose(frames, np.broadcast_to(3+2*time[:,None], frames.shape), atol=1e-12)


@pytest.mark.parametrize('alpha', [.01, .2, 2.])
def test_infinite_heat_gaussian(alpha):
    x, time = np.linspace(-4, 4, 101), np.linspace(0, 2, 21)
    frames = infinite_heat(lambda x: np.exp(-x*x), x, time, alpha, 160)
    width = 1+4*alpha*time[:,None]
    np.testing.assert_allclose(frames, np.exp(-x[None,:]**2/width)/np.sqrt(width), atol=1e-8)


def test_infinite_heat_constant_preservation():
    result = get_topic('infinite_heat').compute({'initial': '3'})
    np.testing.assert_allclose(result.data['frames'], 3, atol=1e-12)


@pytest.mark.parametrize('velocity_only', [False, True])
def test_rectangle_exact_single_mode(velocity_only):
    length, height, speed = 2., 3., .7
    x, y, time = np.linspace(0,length,81), np.linspace(0,height,81), np.linspace(0,2,31)
    basis = np.sin(2*np.pi*x[:,None]/length)*np.sin(3*np.pi*y[None,:]/height)
    initial, velocity = (np.zeros_like(basis), basis) if velocity_only else (basis, np.zeros_like(basis))
    frames, energy, a, v, _ = rectangle_membrane(x,y,initial,velocity,length,height,speed,4,time)
    omega = speed*np.pi*np.sqrt((2/length)**2+(3/height)**2)
    factor = np.sin(omega*time)/omega if velocity_only else np.cos(omega*time)
    np.testing.assert_allclose(frames, factor[:,None,None]*basis, atol=1e-12)
    np.testing.assert_allclose(frames[:,[0,-1],:], 0, atol=0)
    np.testing.assert_allclose(frames[:,:,[0,-1]], 0, atol=0)
    np.testing.assert_allclose(energy, energy[0], atol=1e-12)


@pytest.mark.parametrize('m', [0, 1, 2])
def test_disk_single_mode_and_energy(m):
    radius, speed = 2., .7
    r = np.linspace(0,radius,501)
    theta = np.linspace(0,2*np.pi,65,endpoint=False)
    time = np.linspace(0,1,11)
    root = jn_zeros(m,1)[0]
    basis = jv(m,root*r[:,None]/radius)*np.cos(m*theta[None,:])
    frames, energy, a, v, omega, labels = disk_membrane(r,theta,basis,np.zeros_like(basis),radius,speed,2,m,time)
    exact = np.cos(speed*root/radius*time)[:,None,None]*basis
    # 사다리꼴 방사 적분의 O(Δr²) 오차를 포함합니다.
    np.testing.assert_allclose(frames,exact,atol=9e-6)
    np.testing.assert_allclose(energy,energy[0],rtol=1e-12)
    np.testing.assert_allclose(frames[:,-1],0,atol=0)


def test_disk_invalid_center():
    with pytest.raises(ValueError,match='r=0'):
        get_topic('disk_membrane').compute({'initial':'cos(theta)'})


def test_disk_potential_harmonic():
    r = np.linspace(0,2,31)
    theta = np.linspace(0,2*np.pi,301,endpoint=False)
    field, constant, coefficients = disk_potential(lambda theta: 3+2*np.cos(3*theta),r,theta,2.,8)
    expected = 3+2*(r[:,None]/2)**3*np.cos(3*theta)
    np.testing.assert_allclose(field,expected,atol=1e-12)
    assert constant == pytest.approx(3.)


def test_sphere_dipole_and_constant():
    r, theta = np.linspace(0,2,31), np.linspace(0,2*np.pi,61,endpoint=False)
    field, coefficients = sphere_potential(lambda mu: 2+mu,r,theta,2.,7,8001)
    exact = 2+r[:,None]/2*np.cos(theta)
    np.testing.assert_allclose(field,exact,atol=6e-6)
    refined, _ = sphere_potential(lambda mu: 2+mu,r,theta,2.,7,16001)
    assert np.max(np.abs(refined-exact)) < np.max(np.abs(field-exact))/3.9
    np.testing.assert_allclose(field[0],2,atol=1e-12)


def test_pde_residual_is_not_sample_proof():
    zero = get_topic('pde_basics').compute({})
    assert '잔차 = 0' in zero.data['symbolic']
    nonzero = get_topic('pde_basics').compute({'candidate':'x^2+y^2'})
    assert nonzero.metrics['표본 잔차 최대값'] == pytest.approx(4.)
    sampled = get_topic('pde_basics').compute({'mode':'열','candidate':'t^2','time':0.})
    assert sampled.metrics['표본 잔차 최대값'] == 0
    assert '기호적으로 0: False' in sampled.data['symbolic']


def test_laplace_boundary_causality_and_nonnegative_domain():
    result = get_topic('pde_laplace').compute({'boundary':'sqrt(t)','speed':2.,'end_time':1.})
    d = result.data
    delay = d['time'][:,None]-d['x'][None,:]/2
    np.testing.assert_allclose(d['frames'],np.sqrt(np.maximum(delay,0)),atol=1e-14)
    np.testing.assert_allclose(d['frames'][:,0],np.sqrt(d['time']))


def test_semiaxis_heat_step_and_corner():
    result = get_topic('pde_laplace').compute({'mode':'반무한 열전도','alpha':.5,'amplitude':2.})
    d = result.data
    np.testing.assert_allclose(d['frames'][:,0],2.)
    np.testing.assert_allclose(d['frames'][0,1:],0.)
    np.testing.assert_allclose(d['frames'][-1],2*erfc(d['x']/(2*np.sqrt(.5*d['time'][-1]))))


NEW_IDS = ['pde_basics','dalembert','infinite_heat','rectangle_membrane','disk_membrane','coordinate_potential','pde_laplace']


@pytest.mark.parametrize('topic_id',NEW_IDS)
def test_figures_and_persistence(topic_id,tmp_path):
    from engineering_math.core.project import save_settings,load_settings,export_results
    from engineering_math.topics.lessons import lesson_for
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt
    topic = get_topic(topic_id)
    result = topic.compute({})
    save_settings(tmp_path/'settings.json',topic_id,result.params)
    assert load_settings(tmp_path/'settings.json')['params'] == result.params
    export_results(tmp_path/'result.zip',result)
    assert '12.' in lesson_for(topic_id)
    for figure in topic.figures(result).values():
        FigureCanvasAgg(figure).draw()
        plt.close(figure)


@pytest.mark.parametrize('topic_id',NEW_IDS)
def test_examples(topic_id):
    topic = get_topic(topic_id)
    for params in topic.examples.values():
        result = topic.compute(params)
        assert all(np.isfinite(value) for value in result.metrics.values())


@pytest.mark.parametrize('topic_id,params',[
    ('dalembert',{'initial':'sqrt(x)'}),('infinite_heat',{'initial':'1/0'}),
    ('rectangle_membrane',{'initial':'1/0'}),('pde_basics',{'candidate':'1/0'}),
    ('coordinate_potential',{'boundary':'1/0'}),('pde_laplace',{'boundary':'1/0'})])
def test_bad_inputs(topic_id,params):
    with pytest.raises(ValueError):
        get_topic(topic_id).compute(params)

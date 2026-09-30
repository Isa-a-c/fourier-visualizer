"""교재 11장의 해석식·정규화·경계조건·저장 회귀 검증입니다."""
import numpy as np
import pytest
from engineering_math.core.chapter11 import half_range, project_basis, sturm_basis, orthogonal_basis, integral_transform
from engineering_math.topics import get_topic, get_topics
from engineering_math.topics.chapter11_transforms import PAIRS, exact_pair, ROOT_TWO_OVER_PI
from engineering_math.core.project import save_settings, load_settings, export_results


def test_half_range_x_coefficients():
    length = np.pi
    x = np.linspace(0, length, 12001)
    n = np.arange(1, 11)
    a0, bn, sums = half_range(x, x, length, 10, False)
    assert a0 == 0
    # 사다리꼴 오차는 O(n Δx²)이며 n=10에서 약 1.15e-7입니다.
    np.testing.assert_allclose(bn, 2 * length * (-1.) ** (n + 1) / (n * np.pi), rtol=0, atol=1.3e-7)
    fine = half_range(np.linspace(0, length, 24001), np.linspace(0, length, 24001), length, 10, False)[1]
    exact = 2 * length * (-1.) ** (n + 1) / (n * np.pi)
    assert np.max(np.abs(fine - exact)) < np.max(np.abs(bn - exact)) / 3.9
    a0, an, sums = half_range(x, x, length, 10, True)
    assert a0 == pytest.approx(length / 2)
    np.testing.assert_allclose(an, 2 * length * ((-1.) ** n - 1) / (n * np.pi) ** 2, atol=1e-8)


@pytest.mark.parametrize('boundary', ['DD', 'NN', 'DN'])
def test_sturm_boundary_and_weighted_orthogonality(boundary):
    length = 2.
    x = np.linspace(0, length, 5001)
    basis, eigenvalues = sturm_basis(x, length, 8, boundary, p=2., q=3., r=4.)
    _, gram, _, _ = project_basis(x, x, basis, np.full_like(x, 4.))
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-12)
    start = {'DD': 1., 'NN': 0., 'DN': .5}[boundary]
    assert eigenvalues[0] == pytest.approx((2 * (start * np.pi / length) ** 2 - 3) / 4)
    if boundary != 'NN':
        np.testing.assert_allclose(basis[:, 0], 0, atol=1e-12)
    if boundary == 'DD':
        np.testing.assert_allclose(basis[:, -1], 0, atol=1e-12)
    else:
        derivative = np.gradient(basis, x, axis=1, edge_order=2)
        np.testing.assert_allclose(derivative[:, -1], 0, atol=1e-5)


def test_legendre_polynomial_expansion():
    result = get_topic('orthogonal_series').compute({'function': 'x^2', 'N': 4, 'num_points': 20001})
    np.testing.assert_allclose(result.data['coefficients'], [1/3, 0, 2/3, 0], atol=2e-8)
    assert result.metrics['가중 적분 제곱오차'] < 1e-14


def test_bessel_single_mode_and_weight():
    x = np.linspace(0, 1, 10001)
    basis, weight, eigenvalues = orthogonal_basis('Bessel', x, 5)
    coefficients, gram, sums, _ = project_basis(x, basis[1], basis, weight)
    np.testing.assert_allclose(coefficients, [0, 1, 0, 0, 0], atol=5e-7)
    np.testing.assert_allclose(gram, np.eye(5), atol=5e-7)
    np.testing.assert_allclose(basis[:, -1], 0, atol=1e-14)
    assert np.all(np.diff(eigenvalues) > 0)


def test_minimum_squared_error_perturbation():
    result = get_topic('trig_approximation').compute({'function': 'x^2', 'perturbation': -.7})
    increase = result.metrics['변형 후 제곱오차'] - result.metrics['적분 제곱오차']
    assert increase == pytest.approx(np.pi * .7 ** 2, abs=1e-10)


def test_undamped_resonance_initial_value_solution():
    result = get_topic('fourier_forced').compute({'function': 'sin(x)', 'N': 3,
             'mass': 1., 'stiffness': 1., 'damping': 0., 'end_time': 8.})
    time = result.data['x']
    exact = (np.sin(time) - time * np.cos(time)) / 2
    np.testing.assert_allclose(result.data['state'][0], exact, atol=2e-8)
    assert result.data['particular'] is None
    assert result.metrics['공진 성분 수'] == 1


def test_constant_force_initial_conditions():
    result = get_topic('fourier_forced').compute({'function': '2', 'N': 2,
             'mass': 1., 'stiffness': 4., 'damping': 0., 'end_time': 4.})
    time = result.data['x']
    np.testing.assert_allclose(result.data['state'][0], .5 * (1 - np.cos(2 * time)), atol=1e-8)
    np.testing.assert_allclose(result.data['particular'], .5, atol=1e-12)


def test_continuous_gaussian_and_fft_normalization():
    result = get_topic('continuous_fourier').compute({})
    data = result.data
    np.testing.assert_allclose(data['spectrum'], np.exp(-data['omega'] ** 2 / 2), atol=1e-12)
    np.testing.assert_allclose(data['fft_spectrum'], np.exp(-data['fft_omega'] ** 2 / 2), atol=1e-12)
    assert result.metrics['복원 RMSE'] < 1e-12


def test_translation_phase():
    result = get_topic('continuous_fourier').compute({'function': 'exp(-(x-1)^2/2)'})
    omega = result.data['omega']
    np.testing.assert_allclose(result.data['spectrum'], np.exp(-omega ** 2 / 2 - 1j * omega), atol=2e-12)


@pytest.mark.parametrize('mode', ['코사인', '사인'])
def test_half_line_transform_of_exponential(mode):
    result = get_topic('sine_cosine_transform').compute({'mode': mode, 'function': 'exp(-x)',
              'extent': 16., 'num_points': 8193})
    omega = result.data['omega']
    exact = ROOT_TWO_OVER_PI * (1 if mode == '코사인' else omega) / (1 + omega ** 2)
    np.testing.assert_allclose(result.data['spectrum'], exact, atol=4e-6)


def test_fourier_integral_gaussian():
    result = get_topic('fourier_integral').compute({})
    data = result.data
    np.testing.assert_allclose(data['spectrum'].real, ROOT_TWO_OVER_PI * np.exp(-data['omega'] ** 2 / 2), atol=1e-12)
    np.testing.assert_allclose(data['spectrum'].imag, 0, atol=1e-14)
    assert result.metrics['복원 RMSE'] < 1e-12


@pytest.mark.parametrize('pair', list(PAIRS))
def test_formula_table_against_independent_quadrature(pair):
    from scipy.integrate import quad
    from engineering_math.core.expr import parameter_expression, evaluate
    kind, expression, _, _ = PAIRS[pair]
    a, b = 1.3, .4
    expr = parameter_expression(expression, f'a={a}; b={b}')
    def f(x):
        return float(evaluate(expr, (np.array(x),)))
    for omega in [.3, 1.1]:
        if '펄스' in pair:
            left, right = ((b-a, b+a) if '이동' in pair else (-a, a)) if kind == 'complex' else (0, a)
        else:
            left, right = (-np.inf, np.inf) if kind == 'complex' else (0, np.inf)
        if pair == '복소 · 단측 지수':
            left = 0
        if pair == '복소 · 유리함수':
            real = 2 * quad(lambda x: 1/(x*x+a*a), 0, np.inf, weight='cos', wvar=omega)[0] / np.sqrt(2*np.pi)
            value = real
        elif kind == 'complex':
            real = quad(lambda x: f(x)*np.cos(omega*x), left, right)[0]
            imag = quad(lambda x: -f(x)*np.sin(omega*x), left, right)[0]
            value = (real+1j*imag)/np.sqrt(2*np.pi)
        else:
            wave = np.cos if kind == 'cosine' else np.sin
            value = ROOT_TWO_OVER_PI * quad(lambda x: f(x)*wave(omega*x), left, right)[0]
        assert exact_pair(pair, np.array([omega]), a, b)[0] == pytest.approx(value, abs=2e-8)
    assert np.isfinite(exact_pair(pair, np.array([0.]), a, b)).all()


NEW_IDS = list(get_topics())[1:10]


@pytest.mark.parametrize('topic_id', NEW_IDS)
def test_topic_figures_settings_and_export(topic_id, tmp_path):
    import matplotlib.pyplot as plt
    from engineering_math.topics.lessons import lesson_for
    topic = get_topic(topic_id)
    result = topic.compute({})
    path = tmp_path / '설정.json'
    save_settings(path, topic_id, result.params)
    assert load_settings(path)['params'] == result.params
    export_results(tmp_path / '결과.zip', result)
    assert '11.' in lesson_for(topic_id)
    for figure in topic.figures(result).values():
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(figure).draw()
        plt.close(figure)


@pytest.mark.parametrize('topic_id', NEW_IDS[:-1])
def test_nonfinite_function_input_rejected(topic_id):
    with pytest.raises(ValueError):
        get_topic(topic_id).compute({'function': '1/0'})


def test_invalid_kernel_and_grid():
    with pytest.raises(ValueError):
        integral_transform(np.array([1, 0]), np.ones(2), np.array([0]), 'cosine')
    with pytest.raises(ValueError):
        integral_transform(np.array([0, 1]), np.ones(2), np.array([0]), 'unknown')

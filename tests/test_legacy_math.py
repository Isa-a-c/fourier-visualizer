import numpy as np
import csv
import io
import json
import zipfile
import tempfile
from pathlib import Path
from engineering_math.core.fourier import (parse_expression, FourierSession, sample_function, numpy_function, calculate_fourier_coefficients, calculate_fourier_sum, calculate_errors)
from engineering_math.core.analysis import symmetry, amplitude_phase, transform, alias_signal
from engineering_math.core.diagnostics import integration_stability, convergence_errors
from engineering_math.core.transform_export import export_transform


def test_legacy_spectrum():
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


def test_legacy_diagnostics_export():
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
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'dft.zip'
        export_transform(path,(parse_expression('sin(2*x)'),np.pi,64))
        with zipfile.ZipFile(path) as archive:
            entries=list(csv.DictReader(io.StringIO(archive.read('transform.csv').decode('utf-8-sig'))))
            assert len(entries)==64
            assert np.isclose(float(entries[2]['c_x_imag']),-.5)
            assert np.isclose(float(entries[2]['phase_rad']),-np.pi/2)
            assert entries[0]['phase_rad']==''
            assert json.loads(archive.read('metadata.json'))['M']==64



def test_incremental_legacy_checks():
    for text in ['x','x**2','sin(x)','cos(x)','exp(x)','abs(x)','sign(x)','3']:
        session=FourierSession()
        initial=session.prepare(text,np.pi,5000,10)
        extended=session.prepare(text,np.pi,5000,100)
        assert np.array_equal(initial[4],extended[4][:10])
        assert np.array_equal(initial[5],extended[5][:10])
        expected=calculate_fourier_coefficients(extended[1],extended[2],np.pi,100)
        assert all(np.allclose(a,b,rtol=1e-12,atol=1e-12) for a,b in zip(extended[3:],expected))
        shortened=session.prepare(text,np.pi,5000,5)
        assert len(shortened[4])==5 and len(session.data[4])==100
        changed=session.prepare(text,2.0,1234,20)
        assert len(changed[1])==1234 and np.isclose(changed[1][0],-2)

    # 균일 격자 이외의 표본에서도 공식은 동일해야 한다.
    x=np.r_[-np.pi,np.sort(np.random.default_rng(42).uniform(-np.pi,np.pi,1998)),np.pi]
    y=x**2
    a0,an,bn=calculate_fourier_coefficients(x,y,np.pi,30)
    assert np.isclose(a0,np.trapezoid(y,x=x)/(2*np.pi))
    for n in range(1,31):
        assert np.isclose(an[n-1],np.trapezoid(y*np.cos(n*x),x=x)/np.pi,atol=1e-12)
        assert np.isclose(bn[n-1],np.trapezoid(y*np.sin(n*x),x=x)/np.pi,atol=1e-12)
    expression=parse_expression('sin(2*x)+cos(x)')
    numpy_function.cache_clear()
    sample_function(expression,x)
    sample_function(expression,x)
    assert numpy_function.cache_info().hits==1

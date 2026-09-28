"""증분 계수와 버퍼 재사용이 독립 사다리꼴 적분 결과를 보존하는지 검사한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'fourier_visualizer'))
import numpy as np
from fourier_core import (FourierSession, parse_expression, sample_function,
                          numpy_function, calculate_fourier_coefficients)

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
print('PASS: incremental coefficients, reuse/truncation/invalidation, nonuniform integration, lambdify cache')

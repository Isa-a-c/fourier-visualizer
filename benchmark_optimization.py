"""build/optimization_baseline.py의 변경 전 코드와 동일 조건에서 비교한다."""
import importlib.util
import json
import sys
import time
from statistics import median
from pathlib import Path
root=Path(__file__).parent
sys.path.insert(0,str(root/'fourier_visualizer'))
import numpy as np
import fourier_core as after
spec=importlib.util.spec_from_file_location('before',root/'build/optimization_baseline.py')
before=importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)

def measure(operation,repeats=9):
    operation()
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter()
        operation()
        samples.append(time.perf_counter()-start)
    return median(samples)*1000

x=np.linspace(-np.pi,np.pi,20000)
expression=after.parse_expression('sin(2*x)+cos(x)+exp(x/10)')
a0,an,bn=after.calculate_fourier_coefficients(x,x,np.pi,100)
orders=[1,3,5,10,20,50,100]
rows=[]
for label,old,new in [
    ('sample repeated',lambda:before.sample_function(expression,x),lambda:after.sample_function(expression,x)),
    ('coefficients N=100',lambda:before.calculate_fourier_coefficients(x,x,np.pi,100),lambda:after.calculate_fourier_coefficients(x,x,np.pi,100)),
    ('Gibbs multi-sum',lambda:{n:before.calculate_fourier_sum(x,np.pi,a0,an,bn,n) for n in orders},
                       lambda:after.calculate_fourier_sums(x,np.pi,a0,an,bn,orders)),
]:
    rows.append(dict(case=label,before_ms=measure(old),after_ms=measure(new)))

def extension_time(module):
    values=[]
    for _ in range(9):
        session=module.FourierSession()
        session.prepare('x',np.pi,20000,90)
        start=time.perf_counter()
        session.prepare('x',np.pi,20000,100)
        values.append(time.perf_counter()-start)
    return median(values)*1000
rows.append(dict(case='extend N=90 to 100',before_ms=extension_time(before),after_ms=extension_time(after)))
report=dict(samples=20000,statistic='median of 9 warm runs; numerical functions only, not total UI latency',results=rows)
(root/'build/optimization-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for row in rows:
    print(f"{row['case']}: {row['before_ms']:.3f} -> {row['after_ms']:.3f} ms")

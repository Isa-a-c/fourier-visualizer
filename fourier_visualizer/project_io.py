"""JSON 설정 검증과 CSV 결과 묶음. GUI와 독립적으로 검증한다."""
import csv
import io
import json
import math
import zipfile
from engineering_math.core.fourier import parse_expression

MODES = ['Gibbs 확대', '진폭·위상', 'DFT·FFT·급수 비교', 'Aliasing 실험', '적분 안정성', 'N별 오차 곡선']


def validate_settings(settings):
    if not isinstance(settings, dict) or settings.get('version') != 1:
        raise ValueError('지원하지 않는 설정 파일입니다.')
    parse_expression(settings['function'])
    L = float(parse_expression(settings['L']))
    if not math.isfinite(L) or L <= 0 or not math.isfinite(2*L):
        raise ValueError('L은 유한한 양수여야 합니다.')
    for key, low, high in [('N',1,100), ('num_points',500,20000)]:
        if type(settings[key]) is not int or not low <= settings[key] <= high:
            raise ValueError(f'{key} 범위를 확인하십시오.')
    if not isinstance(settings['comparison_N'], list) or any(type(n) is not int or n not in [1,3,5,10,20,50] for n in settings['comparison_N']):
        raise ValueError('비교 N 목록을 확인하십시오.')
    advanced = settings['advanced']
    if advanced['mode'] not in MODES: raise ValueError('실험 종류를 확인하십시오.')
    M = int(advanced['M'])
    if not 16 <= M <= 256: raise ValueError('M은 16~256이어야 합니다.')
    for key in ['center', 'width', 'frequency', 'rate']:
        if not math.isfinite(float(advanced[key])): raise ValueError('실험 입력은 유한한 수여야 합니다.')
    if not -L <= float(advanced['center']) <= L or float(advanced['width']) <= 0:
        raise ValueError('확대 구간을 확인하십시오.')
    if float(advanced['frequency']) < 0 or float(advanced['rate']) <= 0:
        raise ValueError('실험 주파수를 확인하십시오.')
    orders = [int(v.strip()) for v in str(advanced['orders']).split(',')]
    if not orders or any(not 1 <= n <= 100 for n in orders): raise ValueError('Gibbs 비교 N 범위를 확인하십시오.')
    return settings


def csv_text(headers, rows):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return '\ufeff' + output.getvalue()


def export_results(path, settings, result):
    """마지막 성공 계산과 당시 설정만 내보낸다. CSV는 Excel용 UTF-8 BOM 포함."""
    validate_settings(settings)
    x,y,approximation,error,a0,an,bn,metrics,convergence = result
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('settings.json', json.dumps(settings, ensure_ascii=False, indent=2))
        archive.writestr('samples.csv', csv_text(['x','f(x)','S_N(x)','error'], zip(x,y,approximation,error)))
        archive.writestr('coefficients.csv', csv_text(['n','a_n','b_n'], [(0,a0,0), *[(n,a,b) for n,(a,b) in enumerate(zip(an,bn),1)]]))
        archive.writestr('errors.csv', csv_text(['N','MSE','RMSE','Maximum Absolute Error'],
            [[settings['N'],metrics['MSE'],metrics['RMSE'],metrics['Maximum Absolute Error']]]))
        archive.writestr('convergence.csv', csv_text(['N','MSE','RMSE'], convergence))


def export_transform(path, snapshot):
    """마지막 성공한 DFT 설정으로 원본 FFT 값과 x 기준 위상/계수를 내보낸다."""
    import numpy as np
    from fourier_visualizer.advanced import transform
    expression,L,M = snapshot
    x,y,direct,fast,_ = transform(expression,L,M)
    frequency = np.fft.fftfreq(M,d=2*L/M)
    harmonics = np.fft.fftfreq(M)*M
    corrected = fast/M*np.exp(1j*np.pi*harmonics)
    magnitude = np.abs(corrected)
    phase = np.angle(corrected)
    phase[magnitude <= 1e-10*max(1.,float(np.max(magnitude)))] = np.nan
    rows = [(k,frequency[k],direct[k].real,direct[k].imag,fast[k].real,fast[k].imag,
             corrected[k].real,corrected[k].imag,magnitude[k], '' if np.isnan(phase[k]) else phase[k]) for k in range(M)]
    metadata = dict(function=str(expression), L=L, M=M, endpoint=False,
                    sampling_frequency=M/(2*L), bin_spacing=1/(2*L),
                    frequency_unit='cycles/x', raw_transform='unnormalized forward DFT',
                    coefficient='FFT/M * exp(i*pi*signed_harmonic)',
                    phase='arg(coefficient), radians; blank for near-zero magnitude',
                    spectrum='two-sided magnitude; DC and Nyquist are not doubled')
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('transform.csv',csv_text(['k','frequency','DFT_real','DFT_imag','FFT_real','FFT_imag',
                         'c_x_real','c_x_imag','magnitude','phase_rad'],rows))
        archive.writestr('samples.csv',csv_text(['x','f(x)'],zip(x,y)))
        archive.writestr('metadata.json',json.dumps(metadata,ensure_ascii=False,indent=2))

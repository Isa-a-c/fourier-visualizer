"""학습용 DFT 결과 내보내기입니다."""
import csv
import io
import json
import zipfile

def csv_text(headers, rows):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return '\ufeff' + output.getvalue()


def export_transform(path, snapshot):
    """마지막 성공한 DFT 설정으로 원본 FFT 값과 x 기준 위상/계수를 내보낸다."""
    import numpy as np
    from .analysis import transform
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

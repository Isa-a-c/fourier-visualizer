"""이전 코드와 노트북을 위한 호환 경로. 원본은 engineering_math.core입니다."""
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from engineering_math.core.fourier import (
    X, numpy_function, parse_expression, sample_function, calculate_fourier_coefficients,
    integrate_harmonics, calculate_fourier_sum, calculate_fourier_sums, FourierSession, calculate_errors,
)
from engineering_math.core.plotting import make_comparison_figure, make_error_figure, make_spectrum_figure

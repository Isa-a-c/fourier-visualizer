"""기존 가져오기 경로를 유지하는 호환 모듈입니다."""
from pathlib import Path
import sys
root = str(Path(__file__).resolve().parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
from engineering_math.core.analysis import (symmetry, amplitude_phase, transform, alias_signal, prepare_analysis, make_analysis_figure)

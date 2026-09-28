"""소스 실행과 패키지 빌드를 구분하는 화면 정보."""
import json
import sys
from pathlib import Path

VERSION = '0.5.3'


def build_label():
    if not getattr(sys,'frozen',False): return '개발 소스'
    try:
        metadata=json.loads((Path(sys._MEIPASS)/'build_info.json').read_text(encoding='utf-8'))
        return f"Windows 배포본 · {metadata['built_at']}"
    except (OSError,ValueError,KeyError):
        return 'Windows 배포본'

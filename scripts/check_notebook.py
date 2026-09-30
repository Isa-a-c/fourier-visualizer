"""노트북의 코드 셀을 순서대로 실행하고 원본 소스의 불변성을 검사합니다."""
import sys
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sources = list(root.glob('engineering_math/**/*.py'))
before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
notebook = json.loads((root/'만들기.ipynb').read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        exec(compile(source, f'notebook-cell-{index}', 'exec'), namespace)
assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
print('PASS: notebook cells and source preservation')

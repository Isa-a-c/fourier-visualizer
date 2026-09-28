"""동일 PC에서 PATH/작업 폴더를 분리한 배포 검사. 별도 OS 검사는 아니다."""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

root=Path(__file__).resolve().parent
test_root=Path(tempfile.mkdtemp(prefix='한글 배포 검사 ',dir=root/'build'))
release=test_root/'프로그램 폴더'
shutil.copytree(root/'dist/EngineeringMathStudio',release)
environment=os.environ.copy()
for name in ['PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','TCL_LIBRARY','TK_LIBRARY']:
    environment.pop(name,None)
environment['PATH']=str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32')
environment['MPLCONFIGDIR']=str(test_root/'matplotlib-cache')
report=test_root/'검증 결과.json'
result=subprocess.run([str(release/'EngineeringMathStudio.exe'),'--self-test',str(report)],
    cwd=release,env=environment,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
if result.returncode!=0: raise RuntimeError(f'Portable validation exit={result.returncode}')
payload=json.loads(report.read_text(encoding='utf-8'))
assert payload['passed'] and payload['frozen']
payload['scope']='Same Windows installation; copied Unicode/space path; restricted PATH; no separate OS validation'
(root/'build/portable-validation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: copied Unicode/space path, restricted PATH, frozen subprocesses, animation and export')

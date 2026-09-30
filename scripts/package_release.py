"""빌드 후 README와 라이선스를 포함한 Windows 배포 ZIP을 만든다."""
import hashlib
import importlib.metadata
import shutil
import sys
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from engineering_math import VERSION
folder=root/'dist/EngineeringMathStudio'
if not (folder/'EngineeringMathStudio.exe').is_file():
    raise FileNotFoundError('build_windows.ps1을 먼저 실행해.')
smoke=folder/'smoke-test.json'
if smoke.exists():
    smoke.replace(root/'build/packaged-smoke-test.json')
shutil.copy2(root/'README.md',folder/'README.md')
shutil.copy2(root/'verify_windows.ps1',folder/'verify_windows.ps1')
shutil.copy2(root/'VALIDATION.md',folder/'VALIDATION.md')
(folder/'실행 안내.txt').write_text(
    f'공업수학 학습 스튜디오 v{VERSION}\n\nEngineeringMathStudio.exe를 더블클릭하십시오.\n'
    'Python을 별도로 설치할 필요가 없습니다. _internal을 포함한 전체 폴더를 함께 보관하십시오.\n'
    '주제: 교재 11장 Fourier 해석·12장 PDE, 질량–스프링–댐퍼 ODE, 라플라스 변환\n'
    '학습 설명·계산 설정 탭과 열·파동 비교를 확인하십시오.\n'
    '파일 메뉴: 설정 및 결과 CSV 묶음 저장\n'
    '변화 과정 재생: 차수 또는 시간 애니메이션\n'
    '현재 Windows x64 PC에서 실행 검사 완료. 다른 PC의 동작은 별도 확인이 필요합니다.\n',encoding='utf-8-sig')
license_folder=folder/'licenses'
license_folder.mkdir(exist_ok=True)
packages=['PySide6','PySide6_Essentials','PySide6_Addons','shiboken6','scipy','numpy','matplotlib','sympy','mpmath','pillow','packaging','kiwisolver','contourpy',
          'pyparsing','python-dateutil','six','cycler','fonttools','psutil','charset-normalizer','jinja2','markupsafe']
versions=[]
for package in packages:
    distribution=importlib.metadata.distribution(package)
    versions.append(f'{package}=={distribution.version}')
    for entry in distribution.files or []:
        if any(part.lower().startswith(('license','copying','notice')) for part in entry.parts):
            source=Path(distribution.locate_file(entry))
            if source.is_file():
                target=license_folder/package/Path(str(entry))
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
for name in ['LICENSE.txt','LICENSE']:
    source=Path(sys.base_prefix)/name
    if source.is_file(): shutil.copy2(source,license_folder/f'Python-{name}')
(folder/'versions.txt').write_text('\n'.join(versions),encoding='utf-8')
archive_path=root/f'dist/EngineeringMathStudio-v{VERSION}-windows-x64.zip'
with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for file in folder.rglob('*'):
        if file.is_file() and file.name not in ('validation-report.json', 'validation-report.png', 'wave-small.png', 'ode-window.png', 'laplace-window.png', 'custom-force.png', 'chapter11-window.png', 'chapter12-window.png', 'rectangle_membrane-animation.png', 'disk_membrane-animation.png'):
            archive.write(file,file.relative_to(folder.parent))
digest=hashlib.sha256(archive_path.read_bytes()).hexdigest()
archive_path.with_suffix('.sha256').write_text(f'{digest}  {archive_path.name}\n',encoding='ascii')
print(f'Release ZIP: {archive_path.name}, {archive_path.stat().st_size/1024**2:.1f} MiB')

# 실행 위치와 관계없이 프로젝트 기준으로 Windows 배포 폴더를 생성한다.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Path 'build' -Force | Out-Null
    @{ built_at = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath 'build/build_info.json' -Encoding UTF8
    $fourierBuildInfo = Join-Path $PSScriptRoot 'build/build_info.json'
    & '.\.venv\Scripts\python.exe' -m PyInstaller --noconfirm --windowed --onedir --name EngineeringMathStudio --distpath dist --workpath build --specpath build --paths . --add-data "$fourierBuildInfo;." --exclude-module pandas --exclude-module streamlit --exclude-module IPython --exclude-module pytest --exclude-module tkinter --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --hidden-import matplotlib.backends.backend_qtagg --hidden-import scipy.special main.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
} finally { Pop-Location }
exit 0

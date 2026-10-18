$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$drive = @('Z:', 'Y:', 'X:', 'W:') | Where-Object { -not (Test-Path "$_\") } | Select-Object -First 1
if (-not $drive) {
    throw 'No free temporary drive letter is available for a short-path installation.'
}
& subst.exe $drive $root
if ($LASTEXITCODE -ne 0) {
    throw 'Could not create a temporary short path for Windows package extraction.'
}
$installRoot = "$drive\"

try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Python launcher (py.exe) was not found. Install 64-bit Python 3.12 first.'
    }

    Push-Location $installRoot
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        py -3.12 -m venv .venv
    }

    & '.venv\Scripts\python.exe' -m pip install pip==26.2.1
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not install pinned pip; see output above.'
    }
    & '.venv\Scripts\python.exe' -m pip install -r requirements-lock-win-py312.txt
    if ($LASTEXITCODE -ne 0) {
        throw 'Dependency installation failed; see pip output above.'
    }
}
finally {
    Pop-Location -ErrorAction SilentlyContinue
    & subst.exe $drive /D
}

Write-Host 'Environment is ready. Run .\prepare_models.ps1 to fetch pinned ASR assets, then add an LLM model and run .\run_v2v.bat.'

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $installed = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (-not (Test-Path -LiteralPath $installed)) {
        throw "Python 3.12 is required. Install it with: winget install --id Python.Python.3.12 --exact --scope user"
    }
    & $installed -m venv (Join-Path $PSScriptRoot ".venv")
}
& $python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
& $python -m pip install -e $PSScriptRoot --no-build-isolation
Write-Host "Ready. Run: .\.venv\Scripts\python.exe -m proran experiment"

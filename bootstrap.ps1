$ErrorActionPreference = "Stop"
function Invoke-Python {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $installed = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (-not (Test-Path -LiteralPath $installed)) {
        throw "Python 3.12 is required. Install it with: winget install --id Python.Python.3.12 --exact --scope user"
    }
    & $installed -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment" }
}
Invoke-Python -Arguments @("-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "requirements-dev.txt"), "-r", (Join-Path $PSScriptRoot "requirements-web.txt"))
Invoke-Python -Arguments @("-m", "pip", "install", "-e", $PSScriptRoot, "--no-build-isolation")
Write-Host "Ready. Run: .\.venv\Scripts\python.exe -m web.server"

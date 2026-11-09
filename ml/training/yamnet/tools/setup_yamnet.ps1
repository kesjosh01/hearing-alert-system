param([string]$Python = "")
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $projectRoot ".venv/Scripts/python.exe"
$taskTempPath = Join-Path $projectRoot ".cache/tmp"
New-Item -ItemType Directory -Force -Path $taskTempPath | Out-Null
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
try {
    $env:TEMP = $taskTempPath
    $env:TMP = $taskTempPath
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if (-not $Python) {
            # Prefer a regular CPython install; avoid launching the Store alias.
            $candidates = @(
                (Join-Path $env:LOCALAPPDATA "Programs/Python/Python312/python.exe"),
                (Join-Path $env:LOCALAPPDATA "Programs/Python/Python311/python.exe"),
                (Join-Path $env:USERPROFILE ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe")
            )
            $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        }
        if (-not $Python) { throw "Python 3.11/3.12 not found. Install CPython x64, then pass -Python with its python.exe path." }
        & $Python -c "import sys; assert sys.version_info[:2] in [(3,11),(3,12)], 'Use Python 3.11 or 3.12'"
        if ($LASTEXITCODE -ne 0) { throw "Unsupported Python version" }
        & $Python -m venv (Join-Path $projectRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }
    }
    & $venvPython -c "import sys; assert sys.version_info[:2] in [(3,11),(3,12)], 'Use Python 3.11 or 3.12'"
    if ($LASTEXITCODE -ne 0) { throw "Existing .venv has an unsupported Python version" }
    & $venvPython -m ensurepip --upgrade --default-pip
    if ($LASTEXITCODE -ne 0) { throw "pip initialization failed" }
    & $venvPython -m pip install --cache-dir (Join-Path $projectRoot ".cache/pip") -r (Join-Path $projectRoot "requirements-yamnet.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed; check internet access" }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        throw "FFmpeg missing. Run: winget install --id Gyan.FFmpeg -e ; then reopen PowerShell and retry."
    }
    Push-Location -LiteralPath $projectRoot
    try {
        & $venvPython -m sound_baseline doctor
        if ($LASTEXITCODE -ne 0) { throw "Environment check failed" }
    } finally { Pop-Location }
} finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}

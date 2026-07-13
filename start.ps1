# ISIALAB Annotation Interface — Windows PowerShell launcher
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PythonCmd = $null
foreach ($cmd in @("python", "py", "python3")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        & $cmd -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $PythonCmd = $cmd; break }
    }
}

if (-not $PythonCmd) {
    Write-Host "Error: Python 3.9+ required. https://www.python.org/downloads/" -ForegroundColor Red
    Read-Host "Press Enter"
    exit 1
}

& $PythonCmd scripts/setup.py --start
Read-Host "Press Enter"

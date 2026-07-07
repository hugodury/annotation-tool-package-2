@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="

for %%P in (python py python3) do (
    %%P -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%%P" & goto :found
)

echo Python 3.9+ introuvable — tentative via winget...
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
timeout /t 5 >nul
for %%P in (python py python3) do (
    %%P -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%%P" & goto :found
)

echo Erreur: installez Python 3.9+ depuis https://www.python.org/downloads/
pause
exit /b 1

:found
%PYTHON_CMD% scripts\setup.py --start
pause
endlocal

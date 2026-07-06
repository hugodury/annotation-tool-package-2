@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
for %%P in (python py python3) do (
    %%P --version >nul 2>&1 && set "PYTHON_CMD=%%P" && goto :found
)
echo Error: Python 3.9+ required. https://www.python.org/downloads/
pause
exit /b 1

:found
%PYTHON_CMD% scripts\setup.py --start
pause
endlocal

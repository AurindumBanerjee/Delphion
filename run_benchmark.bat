@echo off
REM One-command retrieval benchmark on Windows.
REM Creates a venv, installs deps, copies books\*.pdf into the eval corpus and runs
REM the before/after sweep. First run downloads ~150MB of models; then ~10-20 min on CPU.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat || goto :fail
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt || goto :fail

if not exist eval\corpus mkdir eval\corpus
copy /Y books\*.pdf eval\corpus\ >nul || goto :fail

python eval\run_eval.py --methods recursive --sizes 500 1000 2000 --k 1 3 5 10 || goto :fail
echo.
echo Done. Results: eval\results\latest.md
goto :eof

:fail
echo.
echo Benchmark failed - copy the error above and send it to Claude.
exit /b 1

@echo off
:: to go cuẻnt 
cd /d %~dp0
cd ..
:: Create python virtual environment
python -m venv test

:: Activate virtual environment
call test\Scripts\activate

:: pip upgrade
python -m pip install --upgrade pip

:: install all needed library
pip install -r requirements.txt

echo Setup done !
pause

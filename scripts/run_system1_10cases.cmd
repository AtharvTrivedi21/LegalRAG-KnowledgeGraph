@echo off
setlocal
cd /d C:\Users\ATHARV\LegalRAG
call venv\Scripts\activate.bat

set SYS3_FAST_MODE=
echo [Run] System 1 started...
powershell -NoProfile -ExecutionPolicy Bypass -Command "python -u -m bns_comparison.run_comparison --systems 1 --cases 1,2,3,4,5,6,7,8,9,10 2>&1 | Tee-Object -FilePath 'bns_comparison/results/run_system1_10cases.txt'"
copy /Y bns_comparison\results\comparison_results.csv bns_comparison\results\comparison_system1.csv > nul

echo [Done] System 1 outputs:
echo   bns_comparison\results\run_system1_10cases.txt
echo   bns_comparison\results\comparison_system1.csv

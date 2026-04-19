@echo off
setlocal
cd /d C:\Users\ATHARV\LegalRAG
call venv\Scripts\activate.bat

set SYS3_FAST_MODE=
echo [Run] System 2 started...
powershell -NoProfile -ExecutionPolicy Bypass -Command "python -u -m bns_comparison.run_comparison --systems 2 --cases 1,2,3,4,5,6,7,8,9,10 2>&1 | Tee-Object -FilePath 'bns_comparison/results/run_system2_10cases.txt'"
copy /Y bns_comparison\results\comparison_results.csv bns_comparison\results\comparison_system2.csv > nul

echo [Done] System 2 outputs:
echo   bns_comparison\results\run_system2_10cases.txt
echo   bns_comparison\results\comparison_system2.csv

@echo off
setlocal
cd /d C:\Users\ATHARV\LegalRAG
call venv\Scripts\activate.bat

call scripts\run_system1_10cases.cmd
call scripts\run_system2_10cases.cmd
call scripts\run_system3_10cases.cmd

echo [Run] Building final summary...
powershell -NoProfile -ExecutionPolicy Bypass -Command "python -m bns_comparison.compare_saved_results 2>&1 | Tee-Object -FilePath 'bns_comparison/results/comparison_separate_runs_summary_log.txt'"

echo [Done] Final summary files:
echo   bns_comparison\results\comparison_separate_runs_summary.txt
echo   bns_comparison\results\comparison_separate_runs_summary_log.txt

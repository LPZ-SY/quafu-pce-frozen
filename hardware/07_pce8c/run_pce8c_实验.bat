@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
rem 2026-10-11 芯片重新标定后重新普查，结果放在 live_1011（10-10 的结果留在 live 里）
set PCE_LIVE=live_1011
cd /d "E:\000量子项目决赛预备\2.QuantumRouteForge：基于精简版增添实验及结果（选择观看）\QuantumRouteForge研究复现包"
".venv\Scripts\python.exe" -u "%~dp0submit_pce8c.py" --run
echo.
echo ===== 脚本已结束，可以关闭此窗口（结果分析由 Claude 在回收后进行） =====
pause

@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PCE_LIVE=live_pce6
cd /d "E:\000量子项目决赛预备\2.QuantumRouteForge：基于精简版增添实验及结果（选择观看）\QuantumRouteForge研究复现包"
".venv\Scripts\python.exe" -u "%~dp0submit_pce6.py" --run --force
echo.
echo ===== 脚本已结束，可以关闭此窗口（结果分析由 Claude 在回收后进行） =====
pause

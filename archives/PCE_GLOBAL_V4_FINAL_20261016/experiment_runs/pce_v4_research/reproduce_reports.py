"""从已保存的原始计数与模型生成报告，不提交真机任务。"""
from pathlib import Path
import sys,subprocess
H=Path(__file__).resolve().parent
for name in ('analyze_hardware.py','audit_delivery.py','report.py','final_research_assessment.py','plot.py'):
    subprocess.run([sys.executable,str(H/name)],cwd=H,check=True)

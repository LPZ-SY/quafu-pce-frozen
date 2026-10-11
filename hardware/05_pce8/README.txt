泡利关联编码真机展示 pce8（2026-10-10）

1. 先看 PREREG.md（做什么、预测什么、怎么解读）。
2. 双击 run_pce8.bat：
   · 读 Baihua、Dongling 当天标定，自动选最好的 8 比特链（结果写在 live\run_log.txt）；
   · 先交 2 个位序诊断，通过后交其余 100 个任务，然后等待收取；
   · 断了就再双击一次，已提交的不会重复提交。
   · 想指定芯片：在命令行运行  python submit_pce8.py --run --chip Baihua
3. 全部完成后告诉 Claude，Claude 读 live 文件夹做分析（analyze_pce8.py）。

文件：
  gen_pce8.py      生成线路与任务清单（已运行，结果冻结在 task_manifest.json）
  submit_pce8.py   提交 / 收取
  analyze_pce8.py  分析
  circuits\        逻辑线路（q[0]–q[7]），提交时换成物理比特，另存 live\physical\
  ideal\           每条线路的理想测量分布
  pce_scale.py、scale_pairs.py、diag_two.py、stress.py  生成和分析所依赖的代码

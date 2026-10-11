泡利关联编码：规模对比（探索性，未预登记）

要回答的问题：规模越大，量子方法能否比经典方法得到更多、更短的路线？

运行（在 disco 上）：
  cd pce_scale_disco
  python3 pce_scale.py 20001 20021 $(nproc) main
  python3 analyze_pce.py pce_scale_main.json > pce_scale_main_analysis.txt

依赖：numpy、scipy、numba（和之前 seg_prereg、seed_refine 用的一样）。
运行中只打印进度；结果写在 pce_scale_main.json。
跑完后把 pce_scale_main.json 和 pce_scale_main_analysis.txt 放回 E:\000量子项目决赛预备\pce_scale_disco\ 即可。

预计耗时：单个实例单核约 35–45 分钟（12 辆车那一档最慢）；20 个实例按核数并行。
  例如 20 核约 45 分钟，8 核约 2 小时。

可调的环境变量：R=随机起点次数（默认 5），STEPS=调参最多步数（默认 1000）。

设置：
  实例 120 客户，需求 1–10，车载量 60；扫描法分车后“搬一个＋换一个”跑到收敛，作为起点。
  同一实例取相邻的 3 / 6 / 12 辆车，比特数 8 / 10 / 12。
  每个客户一个变量：留在原车，或换到这几辆车里离它最近的另一辆。
  五种方法，各 5 个随机起点：量子（纠缠线路）、去掉纠缠门、经典直接调变量、经典模拟退火、随机换车。
  读出后统一处理：超载修正 → 重新排路线 → 算整个实例总路程 → 对这几辆车轻精修 → 再算一次。

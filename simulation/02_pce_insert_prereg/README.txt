泡利关联编码：精确插入损失的机制检验（第一阶段 K=2）
预登记：项目文档《泡利关联编码_精确插入损失_机制检验_预登记》（本目录 PREREG.md 为副本，含运行前补充说明）。

文件
  pce_insert.py      主脚本（子问题、损失、三组优化、枚举、启发式、表达诊断）
  pce_scale.py       实例生成、线路模拟、Adam（沿用项目原文件，未改动）
  scale_pairs.py     路线计算（沿用项目原文件，未改动）
  analyze_insert.py  按预登记汇总
  verify.py / verify3.py / verify.json   实现正确性检查及结果（运行前已通过）
  TUNING_RULE.txt / tuning_result.txt   容量惩罚系数的调定规则与结果
  SHA256SUMS         以上文件的校验值

在 disco 上运行（Python 3，需要 numpy、scipy、numba）
  pip install numpy scipy numba
  sha256sum -c SHA256SUMS
  A_CAP=0.5 EXPR=1 nohup python3 pce_insert.py 2 50001 50031 $(nproc) K2_main > K2_main.log 2>&1 &
  # 进度：tail -f K2_main.log   （每行：已完成个数 种子 累计秒数）
  python3 analyze_insert.py ins_K2_main.json > ins_K2_main_analysis.txt

单核每道题约 3–4 分钟；30 道题、16 核约 10 分钟。
跑完把 ins_K2_main.json、ins_K2_main_analysis.txt、K2_main.log 拷回本目录即可。
不要修改脚本或参数；如需改动，先另写预登记补充说明。

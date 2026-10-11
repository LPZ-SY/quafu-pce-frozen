泡利关联编码：间隔下限训练（已预登记，见项目文档《泡利关联编码_间隔下限训练_预登记》）

要回答的问题：训练时要求每个变量的信号不小于 τ，真机噪声下读出的方案能不能和无噪声时一致？质量损失多少？

运行（在 disco 上，和 pce_scale.py、scale_pairs.py 放在同一个文件夹）：
  cd pce_scale_disco
  python3 pce_margin.py 40001 40011 $(nproc) main
  python3 analyze_margin.py pce_margin_main.json > pce_margin_main_analysis.txt

依赖：numpy、scipy、numba（和 pce_scale.py 一样）。
预计耗时：10 个实例；10 核以上并行约 2–3 小时（12 辆车那一档最慢）。
想先快速看 3、6 辆车：在命令前加 SIZES=3,6

跑完后把 pce_margin_main.json 和 pce_margin_main_analysis.txt 放回 E:\000量子项目决赛预备\pce_scale_disco\ 即可。

泡利关联编码真机展示 pce8c（第二次重跑，2026-10-10）

分两步运行：

第一步  门相位普查：双击 run_pce8c.bat
  · 一次性把三块芯片（Shenglian、Baihua、Dongling）的普查任务全部交上去，不用等；
  · 哪块芯片的结果先回来，就先弹窗、响铃，并把结果写进 live\survey_芯片名.txt；
  · Baihua、Dongling 通常先出结果；Shenglian 排队久，出结果时会再弹窗提醒；
  · 窗口可以一直开着等；中途关掉也没关系，再双击一次会继续收取，已提交的不会重复提交。

第二步  选链 + 正式检测 + 实验：双击 run_pce8c_实验.bat
  · 只使用“普查已完成”的芯片选链。想等 Shenglian 一起比较，就等它的普查结果出来后再运行这一步；
    不想等，也可以先运行，Shenglian 这次就不参与。
  · 想手动限定芯片：在命令行运行  python submit_pce8c.py --run --chips Baihua Dongling

在 disco 服务器上运行（需要已装 quark）：
  1. 把 hw_package_pce8c 文件夹拷到 disco，在文件夹里放 .env，内容一行：QUAFU_API_TOKEN=你的令牌
  2. 第一步：cd hw_package_pce8c && nohup python3 -u submit_pce8c.py --survey > survey.out 2>&1 &
     看结果：cat live/survey_*.txt；看进度：tail -f live/run_log.txt（服务器上没有弹窗）
  3. 第二步：nohup python3 -u submit_pce8c.py --run > run.out 2>&1 &

跑完后把 live 文件夹放回 E:\000量子项目决赛预备\5.真机实验包_未提交\hw_package_pce8c\，告诉 Claude。

# PCE_GLOBAL_V4_FINAL：2026年10月16日代码存档

本目录保存当前最新版代码、冻结模型参数、两项正式实例、训练目标、精修器及实验记录。独立存档分支和日期标签用于恢复；原仓库主分支保持原来的冻结版本。

## 恢复版本

```powershell
git clone --branch pce-global-v4-final-20261016 https://github.com/LPZ-SY/quafu-pce-frozen.git
cd quafu-pce-frozen/archives/PCE_GLOBAL_V4_FINAL_20261016
python verify_archive.py
```

直接克隆日期标签得到固定版本。后续开发应从该标签另建分支。

Git目录保存代码、300个已保存模型的结果文件（含冻结参数）、实例、协议、四张主表、训练曲线、真机线路和已回收计数。完整逐批数据与训练检查点较大，随同一GitHub版本的发布附件 `PCE_GLOBAL_V4_FINAL_20261016_full.zip` 保存；解压到本目录的独立副本即可补齐原相对路径。`archive_manifest.json` 逐文件注明在Git中或仅在完整附件中保存，并给出SHA-256校验值。

## 主要文件

- `experiment_runs/pce_v4_research/core.py`：编码、关联测量及分车解码。
- `experiment_runs/pce_v4_research/final_objective_accelerated.py`：最终可行硬门槛目标和固定精修器；最多评价10000个候选。
- `experiment_runs/pce_v4_research/run_remote_training.py`：在disco使用的最终目标训练入口。
- `experiment_runs/pce_v4_research/PCE_GLOBAL_V4_FINAL.json`、`final_objective_protocol.json`：冻结模型及训练协议。
- `experiment_runs/pce_v4_research/models/`：原样保存的模型参数及结果。
- `experiment_runs/pce_chip_selection_20261016/`：今日选片代码、三芯片线路、台账及最新真机追加分析。
- `experiment_runs/pce_chip_selection_20261016/正式实例完整对照.txt`：最新真机、理想模拟、约束内随机及经典求解的对照。

## 环境及复现

研究计算使用Python 3.12；基础依赖见根目录 `requirements-core.txt`，报告依赖另见原 `requirements-report.txt`。历史文件中的本地环境版本是当时记录，不代表推荐安装组合。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-core.txt
.venv/Scripts/python verify_archive.py
```

完整发布附件解压后，可在安装报告依赖的环境运行：

```powershell
python experiment_runs/pce_v4_research/reproduce_reports.py
```

原始脚本按运行当时状态原样保存。部分远程入口会检查disco的主机名，部分历史重跑入口依赖当时本地布局；恢复后可先验证文件和读取已有参数，不应直接盲跑全部历史入口。原始目录和日期标记保持不变。

真机SDK为可选第三方依赖，版本记录在 `requirements-hardware.txt`；本存档不复制SDK和账号凭据。原真机辅助脚本仍保留原本的本地凭据路径及读取方式，实际重新提交前需在个人环境中配置；离线复核已有结果不需要凭据。所有提交台账一并保存，恢复代码不表示允许重复提交已有任务。

## 结果快照及边界

最新30客户／5车东岭真机：168项任务全部返回。训练与随机参数各3个冻结种子、各24份完整读出，按实际每份6144次测量追加分析。精修后路线分别698.0417、705.4167，均全部最终可行；精修改动客户数18.3750、17.5833，未达到全部改善要求。已知最佳参考值和本轮经典直接求解结果均为672。不是量子优势结论。

今天的45项正式实例任务已提交，最后快照时间为北京时间2026年10月16日11:16:44，均待返回。其中白桦与圣莲覆盖30客户／5车及44客户／7车，东岭覆盖30客户／5车。此存档不会自动更新后续返回状态。

早期报告中“145项返回、23项待回收”和“44客户未提交”等句子是当时快照，原文件留作追溯；当前状态以10月16日追加分析及台账为准。按实际6144次的真机结果不能混入原4096次模拟主比较。

PCE表示能否带来独立优势尚未证实，当前证据支持学到部分任务信息，但真机改善有限，且未超过强经典求解器。

## 存档完整性

源代码逐字节保留，只有原始平台返回计数文件去除无关元数据，保留计数和状态。`archive_manifest.json` 同时记录源文件及存档文件散列值。没有包含`.env`、私钥、平台令牌、缓存或重复传输包。

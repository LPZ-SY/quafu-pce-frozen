"""生成交付索引、环境与哈希清单，并打包科研证据。"""
from core import *
import importlib.metadata,zipfile,datetime
def main():
    status=read(HERE/'delivery_status.json');audit=read(HERE/'delivery_verification.json');assert audit['passed'] and status['tested_models']==300
    env={}
    for name in ('numpy','numba','matplotlib','scipy','ortools','PyYAML'):
        try:env[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:env[name]='未安装'
    save(HERE/'environment_versions.json',dict(local_python=sys.version,local_packages=env,
        remote=dict(python='/tmp/pzlv_search_budget_audit_20261014_r01/.venv/bin/python',numpy='2.0.2',numba='0.60.0'),
        repeated_simulation_check='本地与服务器90份重复完整计数文件、对应测试结果逐项相同。'))
    text='''PCE_GLOBAL_V4_FINAL 本轮正式实验交付

先读 final_conclusion.md；验收状态见 final_acceptance.json。
30客户／5车：20个确认种子，三项改善均通过统计检验。
44客户／7车：10个验证种子，最终路线及原始超载改善，最终精修改动人数增加；未达到全部要求。
两规模主测试训练组精修后均100%可行；两者均未超越所列经典直接求解基线。
剩余模拟在disco完成，原始阶段及部分最终目标训练在本地完成。
真机已提交168项，本次快照145项返回、23项待回收；返回测量数不符冻结预算，未纳入主比较。

四张主表（同时提供CSV及Markdown）：
tables/table_core_model_evolution：首轮、结构修改与最终目标模型演进。
tables/table_training_value：30客户独立确认与所有对照。
tables/table_scale_validation：44客户规模验证与所有对照。
tables/table_hardware：真实任务状态、有效数据及编译理想对照。

图形：figures/training_curves.png及.pdf；figures/final_comparison.png及.pdf。
原始记录：raw_results.jsonl，300模型×3测量档×64批，共57600批。
完整测量计数与分数：final_tests；训练参数、检查点及验证计数：models。
信息对照：final_attribution；映射复验：mapping_sensitivity。
预算：tables/budget_ledger；迁移中止成本：migration.json及migration_interrupted。
实际真机任务编号、编译电路与原始计数：hardware。
完整性核验：delivery_verification.json；重复结果一致性：remote_import_verification.json。
哈希清单：artifact_manifest.json。

补充表包含：配对统计、信息对照、经典对照、精修机制、解码偏向、测量稳定性、容量约束辅助实验、映射敏感性。
统计单位为训练种子，测量重复不充当独立算法重复；仅两个实例，不外推普遍量子优势。

在已安装依赖的环境，从项目根目录重新生成报告：
python experiment_runs/pce_v4_research/reproduce_reports.py

数据计算依赖的共享精修代码随压缩包放在experiment_runs/pce_stable_controls_20261015/shared.py。
环境版本见environment_versions.json；中文绘图默认使用Windows微软雅黑字体。
完整重跑历史旧版还依赖原项目历史模块；交付包支持最终目标模块及已保存证据的报告再生成。
已有结果会被执行脚本跳过；需要重新训练时应在独立目录保留相同输入和协议。
真机提交不包含在报告再生成命令中。
'''
    (HERE/'交付说明.txt').write_text(text,encoding='utf-8')
    (HERE/'requirements-report.txt').write_text('\n'.join(f'{k}=={v}' for k,v in env.items() if v!='未安装')+'\n',encoding='utf-8')
    # 计算环境与完成记录属于交付，传输归档属于重复中间文件。
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.gz','.tmp','.zip') and p.name!='artifact_manifest.json']
    manifest=[dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)]
    shared=HERE.parent/'pce_stable_controls_20261015/shared.py'
    save(HERE/'artifact_manifest.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=manifest,external_dependency=dict(path='../pce_stable_controls_20261015/shared.py',sha256=digest(shared)),
        excluded='重复传输归档、Python编译缓存与临时文件；不含任何平台凭据。'))
    files.append(HERE/'artifact_manifest.json');out=HERE.parent/'PCE_GLOBAL_V4_FINAL_20261015.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files:z.write(p,'experiment_runs/pce_v4_research/'+p.relative_to(HERE).as_posix())
        z.write(shared,'experiment_runs/pce_stable_controls_20261015/shared.py')
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    print('交付包',str(out),'字节',out.stat().st_size,'文件',len(files)+1,flush=True)
if __name__=='__main__':main()

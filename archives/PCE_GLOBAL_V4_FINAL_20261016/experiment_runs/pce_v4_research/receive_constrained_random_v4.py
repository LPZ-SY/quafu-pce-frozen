"""本地只回收与校验文件；新增实验和统计均已在disco完成。"""
from pathlib import Path
import tarfile,hashlib,json,zipfile
H=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    archive=H.parent/'constrained_random_v4_delivery_20261015.tar.gz'
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            if not item.isfile():raise RuntimeError('非普通文件')
            target=(H/item.name).resolve()
            if H not in target.parents:raise RuntimeError('路径越界')
            payload=tar.extractfile(item).read()
            if target.exists():assert target.read_bytes()==payload,item.name
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    O=H/'constrained_random_v4';manifest=json.loads((O/'artifact_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:assert digest(H/item['path'])==item['sha256'],item['path']
    references=[]
    for cohort,name,seeds in [('confirmation','B-n31-k5',range(2001,2021)),('scale','A-n45-k7',range(3001,3011))]:
        for sd in seeds:
            p=H/'final_tests'/cohort/name/f'FINAL_JOINT_{sd}.json'
            data=json.loads(p.read_text(encoding='utf-8'));model=H/data['model']
            assert digest(model)==data['model_sha256']
            references.append(dict(test_file=p.relative_to(H).as_posix(),test_sha256=digest(p),model_file=data['model'],model_sha256=digest(model)))
    (O/'local_import_verification.json').write_text(json.dumps(dict(passed=True,archive_sha256=digest(archive),
        checked_server_files=len(manifest['files']),quantum_references=references,
        note='新增计算与统计均在disco完成；本地仅解包、核对哈希和整理交付。'),ensure_ascii=False,indent=1),encoding='utf-8')
    report=(O/'比较结论.txt').read_text(encoding='utf-8')
    combined=H/'final_conclusion_with_constrained_random.md'
    combined.write_text('# 含约束内随机对照的更新结论\n\n'
        '新增正式对照表明：30客户训练后的量子流程在所有六档约束内随机对照中路线均显著更短；44客户只显著优于单份约束内随机，4096份选优对照则显著优于量子流程。不能将相对无约束随机参数的改善概括为对所有随机方法的优势。\n\n'
        '约束内随机起点已满足容量限制，原始超载为零；当前量子流程不存在精修前可行性优势。相同精修预算不代表相同端到端计算成本。模型参数未因补充比较而调整。\n\n'
        '## 新增约束内随机比较\n\n'+report+'\n## 原交付结论（原样保留，结合上面的新增证据阅读）\n\n'+
        (H/'final_conclusion.md').read_text(encoding='utf-8'),encoding='utf-8')
    guide=O/'交付说明.txt'
    guide.write_text('本补充包与此前PCE_GLOBAL_V4_FINAL完整交付包配套使用。\n'
        '新增实验和统计在disco执行，execution.json及analysis_execution.json记录实际主机。\n'
        '比较结论.txt和tables/table_constrained_random_v4给出六档结果及多重比较校正。\n'
        'results保存30个主种子的原始分配、精修结果、全部候选距离、生成尝试数及随机种子。\n'
        'paired_statistics.json保留逐种子路线差；原量子完整读出在先前完整交付包中，哈希见local_import_verification.json。\n'
        '每档均使用相同的10000候选精修器；构造在容量内随机进行，但不是全可行域均匀抽样。\n'
        '1份为主对照，六档候选数为1、4、16、64、256、4096，全部报告。\n',encoding='utf-8')
    paths=[H/item['path'] for item in manifest['files']]+[O/'artifact_manifest.json',O/'local_import_verification.json',guide,combined]
    out=H.parent/'PCE_GLOBAL_V4_FINAL_约束内随机补充_20261015.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in paths:z.write(p,'experiment_runs/pce_v4_research/'+p.relative_to(H).as_posix())
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    print('回收与哈希核对通过',len(manifest['files']),'补充包字节',out.stat().st_size)
if __name__=='__main__':main()

"""独立测量种子检验同分车信号增强；不提交真机。"""
import run as R
import numpy as np
import zlib

def main():
    D=R.C.read(R.HERE/'hardware/logical_tasks.json')['sources']
    fixed=R.C.read(R.HERE/'stability_results.json');assert len(fixed)==12
    results=[]
    for rec in fixed:
        key=rec['source'];src=D[key];S=R.C.A.setup(src['instance'])
        slots=[(f,tuple(q)) for f,q in src['slots']];c=R.K.CircuitK(8,4,True).prepare(slots)
        actual=c.expect_k(c.run(np.array(rec['theta'])))
        assert np.max(np.abs(actual-rec['outputs']))<1e-12
        assert np.array_equal(actual>0,np.array(src['outputs'])>0)
        assert np.array_equal(R.decode(actual,S['n'],S['K']),src['initial_assignment'])
        result=dict(source=key,instance=src['instance'],min_before=rec['min_before'],min_after=rec['min_after'])
        for label,theta in [('original',src['theta']),('revised',rec['theta'])]:
            pp=c.probs(c.run(np.array(theta)));rng=np.random.default_rng(zlib.crc32(('holdout64:'+key).encode()))
            ok=same=0;gaps=[]
            for r in range(64):
                e=np.zeros(len(slots))
                for f,(idx,signs) in c.groups.items():e[idx]=rng.multinomial(4096,pp[f]/pp[f].sum())@signs/4096
                v=R.decode(e,S['n'],S['K']);d=R.C.A.assess(S,v)
                ok+=d['feasible'];same+=np.array_equal(v,src['initial_assignment'])
                if d['feasible']:gaps.append(d['gap_bks_percent'])
            result[label]=dict(repeats=64,feasible=int(ok),same=int(same),mean_feasible_gap=float(np.mean(gaps)) if gaps else None)
        results.append(result)
    R.save('stability_validation.json',dict(passed=True,unchanged_assignments=12,rows=results,
        note='独立有限测量模拟，不包含真实芯片噪声；未对修正参数执行新真机任务。'))
    lines=['同分车信号增强：disco修正与独立测量验证','',
        '保持原自主训练的12份完整分车全部关联符号不变，8比特4层不变；每条线路追加30000次参数搜索。',
        '优化平均信号裕量缺口，目标0.05；最终仅保留分车不变的参数。目标值不是已实现的最小裕量。',
        '验证使用未用于优化的随机种子：每方向4096次测量，每线路64次独立读出；没有芯片噪声。',
        '这属于已求得分车的读出稳定性修正，不是从经典最优答案拟合线路；其新增计算必须计入成本。','',
        '| 线路 | 原参数可行/64 | 修正后可行/64 | 原参数完整复现/64 | 修正后完整复现/64 |',
        '|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['source']} | {r['original']['feasible']} | {r['revised']['feasible']} | {r['original']['same']} | {r['revised']['same']} |")
    for label in ['original','revised']:
        ok=sum(r[label]['feasible'] for r in results);same=sum(r[label]['same'] for r in results)
        lines.append(f'{label}: feasible {ok}/768, same {same}/768')
        print(label,ok,same,flush=True)
    (R.HERE/'稳定性修正验证.txt').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()

"""从直接优化的搜索结果生成东岭任务：每个（实例, 线路配置）取模拟目标值最低的起点，不按可行性或信号强度筛选。"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pce_k as K

rows = json.loads((HERE / 'search.json').read_text(encoding='utf-8'))
NQ = 8
best = {}
for r in rows:
    key = f"{r['instance']}_{r['kind']}{r['layers']}w{int(r['lam'])}"
    score = (0 if r['assess']['feasible'] else 1, r['proxy'] + r['lam'] * float(np.maximum(0, 0.15 - np.abs(np.array(r['outputs']))).sum()))
    if key not in best or score < best[key][0]:
        best[key] = (score, r)
tasks = [dict(label='zero', qasm=K.qasm_pattern(NQ, '0' * NQ)), dict(label='one', qasm=K.qasm_pattern(NQ, '1' * NQ)),
         dict(label='order', qasm=K.qasm_pattern(NQ, '00101101'))]
sources = {}
for key, (_, r) in best.items():
    sources[key] = {k: r[k] for k in ('instance', 'kind', 'layers', 'lam', 'm', 'theta', 'slots', 'outputs', 'assess', 'proxy', 'frac_ge_03')}
for rep in (1, 2):
    for key, r in sources.items():
        for b in 'ZXY':
            tasks.append(dict(label=f'{key}_{b}_r{rep}', qasm=K.qasm(r['theta'], NQ, r['layers'], b, r['kind'] == 'Q')))
tasks += [dict(label='zero_end', qasm=tasks[0]['qasm']), dict(label='one_end', qasm=tasks[1]['qasm']),
          dict(label='order_end', qasm=tasks[2]['qasm'])]
json.dump(dict(nq=NQ, sources=sources, tasks=tasks, repeats=2), open(HERE / 'logical_tasks.json', 'w'), ensure_ascii=False)
print('sources', len(sources), 'tasks', len(tasks))

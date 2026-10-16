"""四张主表、配对统计、原始记录、训练曲线和科研结论。"""
from core import *
import csv,collections

LABELS={'LEGACY':'旧版两阶段','NEW_DECODER':'新解码','NEW_JOINT':'原始分车联合训练','V42_GEOMETRY':'几何与连续容量改造',
        'FINAL_JOINT':'最终可行硬门槛联合训练','FINAL_RANDOM_SEARCH':'同目标等查询随机参数搜索','RANDOM_SEARCH':'原始目标随机参数搜索',
        'RANDOM_THETA':'未训练随机参数','NULL_READOUT':'无信息读数','SHUFFLED_ROWS':'客户行打乱',
        'CLASSICAL_DIRECT':'经典直接求解','GLOBAL_RANDOM_1':'全局随机1份','GLOBAL_RANDOM_256':'全局随机256份',
        'GLOBAL_RANDOM_4096':'全局随机4096份','LOCAL_RANDOM_MOVE':'局部随机调车','FIXED_8_TWO_VEHICLES_256':'固定8客户双车枚举'}

def write_table(name,rows):
    folder=HERE/'tables';folder.mkdir(exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (folder/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:('NA' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in r.items()})
    def cell(x):
        if x is None:return 'NA'
        if isinstance(x,float):return f'{x:.5g}'
        return str(x).replace('|','／').replace('\n',' ')
    lines=['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']
    lines+=['| '+' | '.join(cell(r.get(k)) for k in keys)+' |' for r in rows]
    (folder/(name+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')

def aggregate_group(group):
    sums=[m['summary']['4096'] for m in group]
    keys=['raw_feasibility_rate','raw_mean_overload','feasible_raw_route_distance','customer_flip_rate',
          'final_feasibility_rate','feasible_final_distance','mean_changed_customers']
    means={k:float(np.mean([s[k] for s in sums if s.get(k) is not None])) if any(s.get(k) is not None for s in sums) else None for k in keys}
    return dict(实例=group[0]['instance'],方法=LABELS.get(group[0]['method'],group[0]['method']),方法编号=group[0]['method'],
        独立种子数=len(group),每模型读出批数=64,每批测量次数=4096,原始可行率=means['raw_feasibility_rate'],
        原始平均超载=means['raw_mean_overload'],原始可行路线均值=means['feasible_raw_route_distance'],
        客户翻车率=means['customer_flip_rate'],最终可行率=means['final_feasibility_rate'],最终可行路线均值=means['feasible_final_distance'],
        平均净跨车客户数=means['mean_changed_customers'],最终失败批数=sum(not r['final']['feasible'] for m in group for r in m['rows'] if r['shots']==4096))

def paired(a,b):
    A={x['seed']:x for x in a};B={x['seed']:x for x in b};rows=[]
    for sd in sorted(A.keys()&B.keys()):
        x,y=A[sd],B[sd];xx={r['batch']:r for r in x['rows'] if r['shots']==4096};yy={r['batch']:r for r in y['rows'] if r['shots']==4096}
        common=[(xx[i],yy[i]) for i in xx.keys()&yy.keys() if xx[i]['final']['feasible'] and yy[i]['final']['feasible']]
        rows.append(dict(seed=sd,raw_overload=x['summary']['4096']['raw_mean_overload']-y['summary']['4096']['raw_mean_overload'],
            changed_customers=x['summary']['4096']['mean_changed_customers']-y['summary']['4096']['mean_changed_customers'],
            final_distance=float(np.mean([u['final']['distance']-v['final']['distance'] for u,v in common])) if common else None,
            final_feasibility=x['summary']['4096']['final_feasibility_rate']-y['summary']['4096']['final_feasibility_rate'],
            common_feasible_batches=len(common)))
    results={};rng=np.random.default_rng(20261015)
    for metric in ('raw_overload','changed_customers','final_distance','final_feasibility'):
        d=np.array([r[metric] for r in rows if r[metric] is not None]);n=len(d)
        if not n:results[metric]=dict(n=0,mean=None,ci95=[None,None],p_two_sided=None);continue
        means=d[rng.integers(n,size=(20000,n))].mean(axis=1);lo,hi=np.quantile(means,[.025,.975])
        signs=rng.choice([-1,1],size=(100000,n));p=(1+np.sum(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-12))/100001
        results[metric]=dict(n=n,mean=float(d.mean()),ci95=[float(lo),float(hi)],p_two_sided=float(p))
    # 三项预设改善要求作为同一检验族，用Holm校正。
    metrics=['raw_overload','changed_customers','final_distance'];ordered=sorted(metrics,key=lambda k:results[k]['p_two_sided'] if results[k]['p_two_sided'] is not None else 1.)
    running=0.
    for i,k in enumerate(ordered):
        p=results[k]['p_two_sided'];running=max(running,min(1.,(3-i)*(p if p is not None else 1.)))
        results[k]['p_holm']=running;results[k]['significant_improvement']=results[k]['mean'] is not None and results[k]['mean']<0 and running<.05 and results[k]['ci95'][1]<0
    return dict(per_seed=rows,effects=results,all_three_supported=all(results[k]['significant_improvement'] for k in metrics),
                note='先在每模型内部汇总测量，再以训练种子配对；路线只用双方同批最终可行的读出；区间反映同实例内种子不确定性。')

def all_data():
    data=[read(p) for p in (HERE/'final_tests').glob('*/*/*.json')]
    groups=collections.defaultdict(list)
    for d in data:groups[(d['cohort'],d['instance'],d['method'])].append(d)
    return data,groups

def report():
    data,groups=all_data();development=[];confirmation=[];scale=[]
    for (cohort,name,method),g in sorted(groups.items()):
        row=aggregate_group(g)
        (development if cohort=='development' else confirmation if cohort=='confirmation' else scale).append(row)
    evolution_order=['LEGACY','NEW_DECODER','NEW_JOINT','V42_GEOMETRY','FINAL_JOINT','RANDOM_THETA']
    development.sort(key=lambda r:evolution_order.index(r['方法编号']) if r['方法编号'] in evolution_order else 99)
    write_table('table_core_model_evolution',development)
    write_table('table_training_value',confirmation)
    write_table('table_scale_validation',scale)
    comparisons=[];detail={}
    for cohort,name in [('confirmation','B-n31-k5'),('scale','A-n45-k7')]:
        a=groups.get((cohort,name,'FINAL_JOINT'),[])
        if not a:continue
        for method in ('RANDOM_THETA','FINAL_RANDOM_SEARCH','V42_GEOMETRY','NEW_JOINT','NEW_DECODER','LEGACY'):
            b=groups.get((cohort,name,method),[])
            if not b:continue
            p=paired(a,b);detail[f'{name}:{method}']=p
            for metric,effect in p['effects'].items():comparisons.append(dict(实例=name,比较=f'最终联合训练减去{LABELS[method]}',指标=metric,**effect))
    save(HERE/'paired_statistics.json',detail);write_table('table_paired_statistics',comparisons)
    ledger=[];curves=[]
    for p in (HERE/'models').glob('*/*/*/result.json'):
        m=read(p);ledger.append(dict(instance=m['instance'],phase=m['phase'],method=m['method'],seed=m['seed'],
            training_seconds=m['training_seconds'],**m['training_ledger']))
        for r in m['trace']:curves.append(dict(instance=m['instance'],phase=m['phase'],method=m['method'],seed=m['seed'],**r))
    write_table('budget_ledger',ledger);write_table('training_curves_source',curves)
    raw=HERE/'raw_results.jsonl'
    with raw.open('w',encoding='utf-8') as f:
        for d in data:
            for r in d['rows']:
                row=dict(r,model=d['model'],model_sha256=d['model_sha256'])
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    info=[];info_groups=collections.defaultdict(list)
    for path in (HERE/'final_attribution').glob('*/*.json'):
        a=read(path)
        for method in ('NULL_READOUT','SHUFFLED_ROWS'):
            rr=[r for r in a['rows'] if r['information']==method and r['permutation']==0]
            if not rr:continue
            sr=summary(rr);sr.update(final_feasibility_rate=float(np.mean([r['final']['feasible'] for r in rr])),
                feasible_final_distance=float(np.mean([r['final']['distance'] for r in rr if r['final']['feasible']])) if any(r['final']['feasible'] for r in rr) else None,
                mean_changed_customers=float(np.mean([r['changed_customers'] for r in rr])))
            for r in rr:r['shots']=4096
            m=dict(instance=a['instance'],seed=a['seed'],method=method,summary={'4096':sr},rows=rr)
            info_groups[(a['instance'],method)].append(m)
    for key,g in sorted(info_groups.items()):info.append(aggregate_group(g))
    write_table('table_information_controls',info)
    for name in ('B-n31-k5','A-n45-k7'):
        cohort='confirmation' if name=='B-n31-k5' else 'scale'
        for method in ('NULL_READOUT','SHUFFLED_ROWS'):
            a=groups.get((cohort,name,'FINAL_JOINT'),[]);b=info_groups.get((name,method),[])
            if a and b:detail[f'{name}:{method}']=paired(a,b)
    save(HERE/'paired_statistics.json',detail)
    classical=[]
    for path in (HERE/'classical').glob('*/*.json'):
        d=read(path)
        for r in d['rows']:
            classical.append(dict(instance=d['instance'],seed=d['seed'],method=r['method'],raw_feasible=r['raw']['feasible'] if r.get('raw') else False,
                raw_overload=r['raw']['overload'] if r.get('raw') else None,final_feasible=r['final']['feasible'] if r.get('final') else False,
                feasible_final_distance=r['final']['distance'] if r.get('final') and r['final']['feasible'] else None,
                native_distance=r.get('native_distance'),candidate_evaluations=r.get('refinement',{}).get('candidate_evaluations'),
                net_changed_customers=r.get('final_changed_customers'),search_seconds=r.get('search_seconds')))
    write_table('table_classical_controls',classical)
    hwfile=HERE/'hardware'/'analysis.json'
    hardware=[]
    for folder in ('hardware','hardware_scale'):
        f=HERE/folder/'analysis.json'
        if not f.exists():continue
        h=read(f)
        for row in h['table']:
            hardware.append(dict(row,已提交任务总数=h['total_tasks'],已返回任务数=h['done_tasks'],待回收任务数=h['unreturned_tasks'],
                协议预算不一致批数=len(h.get('rejected_batches',[])),
                状态='未纳入主比较：实际测量次数与协议不一致' if row['组'].endswith('_QPU') and not row['模型数'] and h.get('rejected_batches') else '按表内有效批次统计'))
    if not hardware:hardware=[dict(状态='未完成真机分析',说明='以hardware目录台账为准；不以模拟替代真机结果')]
    write_table('table_hardware',hardware)
    make_conclusion(groups,detail,hardware,classical)
    save(HERE/'delivery_status.json',dict(model_files=len(ledger),tested_models=len(data),raw_readouts=sum(len(x['rows']) for x in data),
        final_training_confirmation=len(groups.get(('confirmation','B-n31-k5','FINAL_JOINT'),[])),
        final_training_scale=len(groups.get(('scale','A-n45-k7','FINAL_JOINT'),[])),
        random_search_confirmation=len(groups.get(('confirmation','B-n31-k5','FINAL_RANDOM_SEARCH'),[])),
        random_search_scale=len(groups.get(('scale','A-n45-k7','FINAL_RANDOM_SEARCH'),[])),
        expected_final_confirmation=20,expected_final_scale=10,hardware_analysis_present=hwfile.exists()))
    print('报告已生成',len(data),'模型',sum(len(x['rows']) for x in data),'独立读出',flush=True)

def make_conclusion(groups,detail,hardware,classical):
    lines=['# 最终可行硬门槛下的泡利关联编码实验结论','',
        '研究对象为容量约束车辆路径问题（Capacitated Vehicle Routing Problem，CVRP）。泡利关联编码（Pauli Correlation Encoding，PCE）输出完整分车；所有组用完全相同的确定性经典精修，最多评价10000个跨车候选。',
        '本轮实际收到第三版执行书及第四版配置；用户后续明确要求以精修后可行路线为主目标。因此原始分车优先的首轮与一次结构改造保留作历史对照，最终模型重新采用训练中参与精修的目标。不存在可核验的第四版A—F文字，未杜撰其标签。','',
        '最终不可行一律记失败，训练惩罚为1000000加最终超载项；其路线不参与路线优劣或可行均值。可行损失由归一化最终路线、原始超载、净跨车客户数和有限测量翻车率构成。所有组共用精修算法、候选评价上限、停止条件及配对主种子。精修本身没有随机数。',
        '精修前结果（RAW）与统一精修后结果（FINAL）分别报告。净跨车客户数指最终分配与原分配中车辆编号不同的客户数，不是搜索过程累计移动次数。客户翻车率指有限测量分配相对同参数精确概率分配发生变化的客户比例。协议公式中的Q为实例总需求，路线归一化尺度为（客户数＋车辆数）乘最大两点距离。','',
        '## 主结果','']
    for cohort,name in [('confirmation','B-n31-k5'),('scale','A-n45-k7')]:
        a=groups.get((cohort,name,'FINAL_JOINT'),[]);b=groups.get((cohort,name,'RANDOM_THETA'),[])
        if not a or not b:lines.append(f'{name}：最终独立测试尚未完成。');continue
        aa,bb=aggregate_group(a),aggregate_group(b);p=detail[f'{name}:RANDOM_THETA']
        lines.append(f"{name}：{len(a)}个训练种子，每模型64批、每批三基合计4096次测量。训练组最终可行率{aa['最终可行率']:.2%}，随机参数组{bb['最终可行率']:.2%}；最终可行路线均值分别为{aa['最终可行路线均值']:.3f}与{bb['最终可行路线均值']:.3f}。")
        lines.append(f"精修前平均超载分别为{aa['原始平均超载']:.3f}与{bb['原始平均超载']:.3f}；平均净跨车客户数分别为{aa['平均净跨车客户数']:.3f}与{bb['平均净跨车客户数']:.3f}；客户翻车率分别为{aa['客户翻车率']:.2%}与{bb['客户翻车率']:.2%}。")
        for metric,label in [('raw_overload','原始超载'),('changed_customers','净跨车客户数'),('final_distance','最终可行路线')]:
            e=p['effects'][metric];lines.append(f"训练减随机参数的{label}配对差为{e['mean']:.4f}，95%种子配对区间[{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}]，多重检验校正后概率值{e['p_holm']:.5g}。")
        lines.append('三项预设改善条件均获得统计支持。' if p['all_three_supported'] else '三项预设改善条件尚未全部获得统计支持；不能把部分改善写成整体成功。')
        for other in ('FINAL_RANDOM_SEARCH','NEW_JOINT','V42_GEOMETRY'):
            key=f'{name}:{other}'
            if key in detail:
                e=detail[key]['effects']['final_distance'];lines.append(f"相对{LABELS[other]}的共同可行配对路线差为{e['mean']:.4f}，95%区间[{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}]；其他指标见配对统计。")
        cr=[r['feasible_final_distance'] for r in classical if r['instance']==name and r['method']=='CLASSICAL_DIRECT' and r['final_feasible']]
        if cr:
            cmean=float(np.mean(cr));gap=100*(aa['最终可行路线均值']-cmean)/cmean
            lines.append(f"经典直接求解器经同一精修后的平均路线为{cmean:.3f}，最终联合训练相对它的距离差为{gap:+.2f}%。其原生路线另列，经典搜索时间使用前一冻结阶段的实际时间预算，未伪称与新精修目标训练总成本相等。")
            if gap>0:lines.append('本实例中没有超越该经典直接求解基线，不能把相对随机参数的改善表述为实用求解优势。')
        for other in ('NULL_READOUT','SHUFFLED_ROWS'):
            key=f'{name}:{other}'
            if key in detail:
                e=detail[key]['effects']['final_distance'];lines.append(f"相对{LABELS[other]}的共同可行配对路线差为{e['mean']:.3f}，95%区间[{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]；此对照用于检查分数与具体客户的对应是否有用。")
        lines.append('')
    lines+=['## 科研解释与边界','',
        '新目标检验的是量子参数能否学习有利于固定经典精修器的起点。最终收益属于量子表示与经典搜索的联合流程；精修参与每次训练评价所消耗的经典工作已经计账，不能称为免费后处理，也不能据此宣称量子加速。',
        '只有两个指定公开实例。20或10个训练种子和每模型64次测量均不是独立问题实例，统计区间只反映这些实例内的随机性。最终路线比较使用双方共同可行的批次，同时完整报告最终失败率。客户行置换的20次重复嵌套在父模型中。',
        '同预算随机参数搜索采用相同精修目标、相同4096次参数查询与每查询4份完整读出。旧版保留20,001次分车评价及30,001次固定符号稳定性评价；旧版不是等预算单因素消融。',
        '原始阶段为满足第三版协议还保留容量约束分配辅助实验。该阶段含额外经典约束求解，不能冒称直接量子分车，且不进入新训练目标。','',
        '## 真机证据','',
        '真机请求为模拟训练后冻结参数的前三个种子、每参数8批，每批三基合计4096次测量。已提交168个任务，当前保存145个返回结果，23个待回收。用户已要求不等待剩余真机任务。',
        '已回收科学任务每基实际返回2048次，原请求为1365、1366、1365次；位序校验请求256次却返回1024次。它们不符合冻结的测量预算，原始数据全部保留，未纳入主比较；这不等于算法在真机上失败，也不支持真机优势结论。44客户真机本轮未提交。',
        '独立编译模拟器与训练器的量子比特位序不同；分析现已显式转换，并逐模型核验理想概率一致。第四表的理想对照仅是模拟，不当作设备结果。设备、编译、布局、任务编号及预算校验详见hardware目录。','',
        '## 直接相关文献及本轮采用范围','',
        '1. [Sciorilli等，2025，Nature Communications](https://www.nature.com/articles/s41467-024-55346-z)：提供PCE及经典局部改进的基础。本文是最大割问题，不能把其性能直接外推为容量约束车辆路径问题优势。本轮借鉴表示与分阶段记账，不借用其问题结论。',
        '2. [Usuki等，2026年9月预印本](https://arxiv.org/abs/2609.09718)：区分表达能力与训练目标造成的优化困难。本轮直接对固定精修后的真实目标训练，检验优化目标不一致是否限制最终路线；不使用已知好解标签来训练量子参数。',
        '3. [Padín-Martínez等，2026年9月修订预印本](https://arxiv.org/abs/2602.17479v6)：在预算约束最小割中通过逐步二值化改善可行性。本轮原始阶段试验连续容量与几何项；用户更新目标后，最终可行性由相同精修器和失败硬惩罚联合保证。论文的88%—100%可行率不作为本项目证据。',
        '4. [Boussahi，2026年10月14日预印本](https://arxiv.org/abs/2610.19544)：明确有限测量预算与目标锐化、设备噪声之间的关系。本轮采用完整位串共同测量、训练时有限测量反馈和编译后同参数理想对照。论文针对符号最大割的锐化公式不能不加推导地用于本项目多车辆最大分数解码。',
        '5. [Hadfield，2026年10月修订预印本](https://arxiv.org/abs/2608.25151v2)：比特压缩可能转移成本到关联间隔与信息恢复。本轮因此分别报告量子比特数、测量次数、参数查询和精修候选数；并不把特定马约拉纳编码的最坏间隔公式直接套到本映射。','',
        '上述2026年论文均按预印本处理，本轮实验不构成对其全部结论的复现。','',
        '## 文件与复现','',
        '四张主表位于tables目录；raw_results.jsonl保存统一测试的逐次原始与最终分配；完整测量计数位于final_tests的压缩数组文件，参数与训练检查点位于models；训练曲线与原始曲线表位于figures及tables。',
        '原始目标阶段与部分最终目标训练在本地运行；剩余正式训练和补充测试在用户授权的disco运行，服务器实际主机名为lenovo，执行记录见disco_execution.json。加速实现保持整数对称距离下的候选顺序和判定，与40份完整正式规模精修结果逐项一致。中止迁移成本保存在migration.json及migration_interrupted，不伪称无损续跑。',
        'run_all.py all对应原始目标阶段；run_final_training.py all对应用户更新目标阶段；等价加速版本为run_remote_training.py。final_evaluation.py负责统一测试和信息对照，report.py及plot.py生成交付。既有结果会跳过，完整复现应复制到新的空结果目录并使用相同实例与协议。真机操作另行处理，不因重新生成报告而自动提交。']
    (HERE/'final_conclusion.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':report()

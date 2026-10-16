"""由保存的数据生成中文结论与可复核图表，不改变实验结果。"""
import os
from pathlib import Path
import json
import csv
import hashlib
import re
import numpy as np

HERE=Path(os.path.abspath(__file__)).parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'plot_cache'))


def read(path):return json.loads((HERE/path).read_text(encoding='utf-8'))


def main():
    e=read('results/examples.json')['lexicographic']
    rep=read('results/representation_summary.json')
    train=read('training/evaluation.json')
    diagnostics=read('results/objective_diagnostics.json')
    h1=read('hardware/analysis.json')
    h2=read('hardware_refined/analysis.json')
    sources=read('hardware_refined/manifest.json')['sources']
    text=[]
    def line(s=''):
        # 路径与代码文件保留原名，并与中文说明区分。
        s=re.sub(r'https?://\S+|(?:/[A-Za-z0-9_.-]+){2,}|[A-Za-z0-9_./-]+\.(?:py|json|csv|npz)',
                 lambda m:m.group(0) if m.group(0).startswith('http') else '`'+m.group(0)+'`',s)
        text.append(s)
    line('PCE-CVRP：直接编码的决定性实验、模型修正与两轮真机复验')
    line('日期：2026-10-14；计算服务器：disco（主机 lenovo）；真机：白桦。')
    line('泡利关联编码（Pauli Correlation Encoding，PCE）；容量约束车辆路径问题（Capacitated Vehicle Routing Problem，CVRP）。')
    line()
    line('一、结论与项目主线')
    line('原提出的固定算符替换 a=(I−ZiZj)/2 不保持完整分配表示能力，本轮否定的是这种具体等价替换，不是否定 PCE。')
    line('改为 a=(1−sgn〈ZiZj〉)/2 后，7 比特可物理表示 10 客户、2 车辆的全部 1024 种唯一归属分配。符号函数 sgn 根据正负返回 +1 或 −1。该结论是已知分配的可表示性，不是求解或高效制备保证。')
    line('直接优化分配目标的 88 次训练全部得到可行分配，其中 50 次为量子线路模拟、38 次为经典连续优化。后续两轮真机检验不做容量修复。路线质量仍弱于本轮经典对照；没有证明量子优化优势。')
    line('研究继续围绕压缩表示与直接分配目标，不回到量子随机扰动辅助经典搜索。必须明确：本轮修正方案优化的是关联期望值的非线性函数，不是原先那一个固定单份量子态哈密顿量的基态。')
    line()
    line('二、完成的实验与统计单位')
    line('算符检验：10 个十客户基础实例、4 个十六客户基础实例，各用总容量/总需求约 1.02、1.15、1.35 三种水平，共 42 个实例配置。每配置检查 81 个映射，共 3402 行。')
    line('81 个映射包括 1 个按字典序映射、64 个随机映射、16 个保证至少一种唯一归属分配的构造映射；构造过程不读取需求、位置或最优路线。容量向上取整，实际比值见实例文件。')
    line('结构扩展：10、16、24、40、60 客户，每规模 289 个映射，共 1445 个映射，采用二元域消元精确计算唯一归属解数。随机和构造映射是同一实例上的相关对照，不作为独立实例伪重复。')
    line('训练检验：10 客户 6 个实例；16 客户 4 个；24、40、60 客户各 3 个。每方法两次重启，分别使用 4、8 层线路；深度和随机起点共同变化，不能单独归因深度效应。全部车辆数均为 2。')
    line('数据为固定随机种子的合成欧氏距离实例，未经筛选；本轮没有使用标准库实例，不能外推到多车辆或全部 CVRP 分布。')
    line()
    line('三、第 4.2 步的具体结果')
    inst=e['instance'];ground=e['ground_states'][0]
    line(f"十客户代表实例：种子 {inst['seed']}；需求 {inst['demand']}；容量 {inst['capacities']}；总需求 {sum(inst['demand'])}。坐标保存在 results/instances/n10_s100_r1.15.json。")
    line('映射：按客户优先、车辆次优先，将 a_i,k 对应到 7 比特全部两体 Z 对的前 20 对；完整表见 results/lexicographic_mapping.csv。')
    line(f"原模型：{e['original_terms']} 个非零项（含恒等项），最大泡利阶数 {e['original_max_order']}。二体变量的平方惩罚只产生至多四体项；原公式不会自行产生更高阶项。")
    line(f"最低能量 {ground['energy']}；基态编号 {[r['state'] for r in e['ground_states']]}；q0 在左的比特串 {[r['bits_q0_first'] for r in e['ground_states']]}。")
    line(f"解码依次为 {ground['assignment']}；每客户的两位依次代表车辆 1、2。三名客户重复归属、两名客户未归属，唯一归属惩罚为 5；两车载荷均为 38，容量等式惩罚为 0；分配不可行。")
    line('本实例的完整分配空间实际有 426 个容量可行分配，最优总路程 351.686962119181；因此失败不能归因于实例无解。')
    line('全部 128 态的能量、载荷、解码与可行性见 results/lexicographic_all_states_sorted.csv；另一种构造映射的同类文件以 conditioned_000 开头。')
    line('按字典序映射在全部 42 个配置均不能表示唯一归属分配；改容量惩罚或加大惩罚权重也无法创造不存在的表示。十客户随机映射的 1920 个配置只有 35 个包含容量可行分配；构造映射为 184/480，但平均只表示 1 种唯一归属分配。')
    line()
    line('四、为何固定算符模型失败，怎样修正')
    line('单次测量时，关联值为 z_i z_j，满足沿任意闭环乘积为 1。全体 z 同时变号不改变任一两体关联；7 比特最多有 2^6=64 种不同关联模式，而两车辆十客户已有 2^10=1024 种有标签的唯一归属分配。64 是上界，不是 64 个可行分配；本轮字典序映射的可行数量实际为零。')
    line('忽略相同车辆的编号对称性仍有 512 种分组，依旧大于此模型的 64 模式上限。一般固定单次测量解码也不能将超过 128 种确定结果无损放进 7 比特的 128 个基态。这个计数论证不适用于多次测量的期望值符号。')
    line('容量应为 λC Σk max(0,Σi d_i a_i,k−Qk)^2，或使用计入资源的松弛变量。原式 (load−Q)^2 惩罚未满载，容易通过漏分、重复分配填满车辆。')
    line(f"本轮将正确超载能量表精确展开，代表映射得到 {e['inequality_terms']} 项、最高 {e['inequality_max_order']} 体。此能量表法需要枚举 2^q 态，只是精确对照，不是可扩展编译方法。")
    line('〈A〉〈B〉通常不等于〈AB〉，F(〈P〉)通常不等于〈F(P)〉。独立核验中，均匀叠加态的十个唯一归属平方惩罚的算符期望为 5，而先取期望再计算惩罚为 0。混淆两者会改变优化问题。')
    line('修正后的实际损失采用 a=(1−tanh(α〈P〉))/2，并把 α 依次从 2 提升到 5、10、20、40。两车辆互补编码只训练 a_i,1，令 a_i,2=1−a_i,1；这样无需唯一归属惩罚，也不会由符号错误产生双归属。超载仍可能出现，必须单独检查。')
    line('训练目标包含同车客户之间的归一化距离和超载惩罚；同车距离之和是分组目标，不能等同于最短车辆路线。所有路线评估均固定训练所得分配，未跨车修复。')
    line('如必须保留“直接哈密顿量”，需另行构造并核算多份量子态、辅助量子比特等资源；不能把非线性期望值损失重新命名为等价固定哈密顿量。状态相关的有效算符可计算梯度，但其系数随状态更新，不给出固定基态等价性。')
    line()
    line('五、表示能力与测量代价')
    line('对固定 20 个两体 Z 观测量，用线性规划逐一寻找 1024 个目标分配的物理概率分布，使所有关联符号正确且最小绝对值尽可能大。所有目标均成功；最小最优间隔为 0.2，中位数和最大值约 1/3。')
    line('完整概率分布与目标符号保存在 results/representation_states.npz。以各概率的平方根作纯态振幅可直接核验这些 Z 期望值；不保证这些态都能由本轮浅层线路制备。')
    line('更一般地，对 M 个不同非恒等 Z 字符 χ_j(z)，任意目标符号 s_j 都可用 p(z)=2^(−q)[1+(1/M)Σj s_jχ_j(z)] 构造非负概率分布，得到〈P_j〉=s_j/M。这解释了为何期望值符号不受单次测量模式数限制。')
    line('这是数学表示构造，需要已知目标，不是搜索算法；对角概率模型也可以由经典系统表示，不证明量子专属的信息优势。')
    line('理想独立重复测量下，最小间隔 γ 的全变量符号错误概率可用 M exp(−Sγ²/2) 上界控制，其中 S 是测量次数。压缩必须同时报告间隔、测量次数、线路门数与经典计算，不能只报告变量数/比特数。')
    line('公平比特基线：两车辆十客户的 20 个独热变量只有 10 个独立二进制决策。7 比特相对该基线为 10/7≈1.43 倍；互补编码使用 5 比特，相对基线为 2 倍。60 客户使用 12 比特，相对 60 个独立决策为 5 倍；相对 120 个冗余独热变量的 10 倍仅是名义比例。')
    line()
    line('六、训练质量；可行不等于优化优越')
    for r in train['summary']:
        name={'classical':'经典连续优化','complement':'互补关联编码','redundant':'双变量关联编码'}[r['model']]
        gap=r['conditional_mean_exact_route_gap_percent']
        line(f"{r['n']} 客户，{name}：{r['feasible']}/{r['runs']} 次原始可行；量子比特 {r['q']}；平均目标评价 {r['mean_calls']:.1f} 次；"+
             (f"精确路线相对最优差距均值 {gap:.2f}%。" if gap is not None else f"共同路线整理方法下平均距离 {r['mean_heuristic_distance']:.2f}。"))
    line('10、16 客户的全局路线参考枚举所有分配，并对每个客户子集用精确动态规划求最短回路；与六客户全部排列的独立核验误差为零。24 客户以上仅采用相同最近邻和二边交换，没有最优性证明。')
    line('各模型使用相同连续优化器及每阶段迭代上限，实际评价次数、参数量和耗时不同，因此这些是透明的性能参照，不是严格相同总计算预算的优势检验。')
    for n in (10,16):
        subset=[r for r in diagnostics if r['n']==n]
        line(f"进一步把分组目标本身精确求到全局最优：{n} 客户的路线相对最优差距平均仍为 {np.mean([r['proxy_optimal_route_gap_percent'] for r in subset]):.2f}%，最大 {max(r['proxy_optimal_route_gap_percent'] for r in subset):.2f}%。因此同时存在目标不一致和量子训练未找到高质量分组两种问题，详见 results/objective_diagnostics.json。")
    line()
    line('七、第一轮真机结果')
    line(f"29 个任务：5 个位序校验和 8 个方案×3 次重复；每任务 4096 次测量，共 118784 次。全部任务完成，位序校验全通过；{h1['feasible']}/{h1['total']} 次原始分配可行，{h1['identical']}/{h1['total']} 次与理想分配完全一致。")
    for label in dict.fromkeys(r['source_label'] for r in h1['rows']):
        rows=[r for r in h1['rows'] if r['source_label']==label]
        line(f"{label}：可行 {sum(r['result']['feasible'] for r in rows)}/3；完全一致 {sum(r['identical_assignment'] for r in rows)}/3。")
    line('8 个方案按固定实例编号选取，各实例从两次训练重启按离散分组目标选优；没有根据真机成绩换参数。未使用读出缓解或容量修复。原始计数和任务编号分别保存在 hardware/raw 与 hardware/ledger.json。')
    line()
    line('八、根据真机结果修正，再上真机复验')
    line('对第一轮未能完全恢复的 3 个方案，保持量子优化已经找到的客户分配不变，分别以 4、8 层线路进行监督表示训练，增大最弱关联的符号间隔。六种修正均保持原分配。它是读出稳健性修正，不是额外六次独立求解。')
    line('第二轮把三条原线路与六条修正线路同批重新运行，每条 3 次重复，加两个位序校验，共 29 个任务、118784 次测量。原线路为设备随时间变化的同批对照；修正方案根据第一轮失败结果产生，因此不能当作独立泛化样本。')
    line('第二轮针对第一轮的失败个案，27 次读出中 8 次可行、3 次完全恢复。此成功率不能与第一轮全部方案的 17/24 直接比较。应比较下表中同一实例的原线路与修正线路。')
    for label in dict.fromkeys(r['source_label'] for r in h2['rows']):
        rows=[r for r in h2['rows'] if r['source_label']==label]
        src=sources[label]
        line(f"{label}：理想最小间隔 {src['result']['min_margin']:.6f}；可行 {sum(r['result']['feasible'] for r in rows)}/3；恢复最初分配 {sum(r['original_identical'] for r in rows)}/3。")
    line('两轮合计 58 个真机任务、237568 次测量。每条线路均独立模拟编译后的物理门，并核对输出概率最大误差小于 10^−7；只验证了无噪声编译语义，没有据此假设真实硬件无误差。')
    line('决定性发现：16 客户的 4 层修正版恢复原分配 3/3，同批原线路与 8 层修正版均为 0/3。60 客户的 4 层修正版容量可行 3/3，却恢复原分配 0/3；10 客户失败个案仍未解决。增大理想间隔不是充分条件，较深线路的真实误差可能抵消收益。不能把“可行但变了分配”算作完整恢复。')
    line('硬件原始分配对应的最终路线已逐一恢复并核验，见 results/hardware_routes.json。可行分配只在车内排序；不可行分配保持失败状态，路线和距离为缺失值，而不是偷偷修复后填入成绩。')
    line()
    line('九、下一轮科研决策')
    line('优先修正模型而非增加固定哈密顿量的优化深度：继续研究关联期望值符号的直接分配优化，并把“固定哈密顿量等价压缩”作为尚未实现且需单独证明的目标。')
    line('下一项决定性比较应是：同一目标分配下的最小关联间隔、所需线路深度、真实物理门质量与读出成功率；同时把分组目标与实际路程的偏差纳入分析。本轮经典对照路线更短，不能把全部瓶颈都归因于硬件。')
    line('将当前二车辆模型扩展到多车辆时，必须重新构造唯一归属表示；不能直接沿用 a_i,2=1−a_i,1，也不能把二车辆结果包装成已解决一般 CVRP。')
    line('本轮采用全 Z 关联，测量成本便于审计，但对应经典概率分布；后续是否引入不同方向的非对易关联，应以可表示目标、稳健间隔和训练收益的实测比较决定，不因关联数量更多就直接认定更优。')
    line()
    line('十、直接相关文献及可借鉴之处（检索至 2026-10-14）')
    line('1. Sciorilli 等，Nature Communications 16, 476（2025-01-08），Towards large-scale quantum optimization solvers with few qubits。https://doi.org/10.1038/s41467-024-55346-z')
    line('基础定义把二进制信息放在泡利期望值的符号中，优化连续松弛损失；不能据此把期望值乘积换成算符乘积。该论文不在最近一年窗口内，但属于本项目必须对齐的基础工作。')
    line('2. Padín-Martínez 等，Progressive Binarization - Pauli Correlation Encoding: a Continuation Method for Constrained Optimization，2026-09-18 第六版。https://arxiv.org/abs/2602.17479v6')
    line('针对预算约束最小割，提出逐步强化二值化并延续优化；摘要报告用 9 比特处理至多 300 变量，约束满足率 88–100%。本轮借鉴连续到离散的逐阶段训练思路，没有照搬其最小割结论为 CVRP 证据，也未宣称复现其完整算法。')
    line('3. Usuki 等，Enhancing Pauli Correlation Encoding for quantum optimization via systematic expressivity analysis，2026-09-09 预印本。https://arxiv.org/abs/2609.09718v1')
    line('区分给定好解能否被表示与无监督目标能否找到好解；采用监督表示检验与多阶段训练分析最大割。本轮对 1024 种分配的表示检验、88 次直接优化和保持分配的间隔训练分别报告，避免混淆表示、优化与读出。其最大割结论不能直接证明 CVRP 的可训练性。')
    line()
    line('十一、复现与文件索引')
    line('服务器目录：/mnt/home/pzlv/pce_assignment_decisive_20261014')
    line('Python 环境：/mnt/home/pzlv/pce_hardware_capacity_20261011/.venv/bin/python')
    line('pauli_z.py：精确有理数的 I/Z 多项式加乘、约消、高阶展开及白名单表达式编译。未声称实现一般 X/Y 非对易泡利代数。')
    line('run_decisive.py：实例、映射、全部能量表、精确覆盖率、全部 1024 个目标的线性规划、理想有限测量。')
    line('train_assignment.py、analyze_training.py：直接分配优化、经典对照、路线质量与模拟读出。')
    line('hardware_assignment.py、refine_readout.py：两轮真机编译、任务、回收、保持分配的间隔修正。')
    line('verify_independent.py：精确路线与独立排列核验、编译模拟器校验、两种期望值运算的反例。')
    line('results/compiler_verification.json：680 个状态的精确编译核验，最大误差零；training/gradient_verification.json：梯度有限差分最大绝对误差小于 10^−7。')
    line('results/operator_batch.csv、results/structural_scaling.csv、training/evaluation.json、hardware/analysis.json、hardware_refined/analysis.json：完整未筛选结果。')
    line('硬件脚本提交操作会创建新任务；已有台账阻止重复提交，不应删除台账后重跑。运行分析与报告脚本不会提交任务。')
    (HERE/'实验结论与模型修正.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    draw(train,h1,h2)


def draw(train,h1,h2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fontpath=Path('C:/Windows/Fonts/msyh.ttc')
    if fontpath.exists():
        font=FontProperties(fname=str(fontpath)).get_name()
    else:
        font='Noto Sans CJK SC'
    colors={'classical':'#555555','complement':'#0072B2','redundant':'#B34D00'}
    names={'classical':'经典连续优化','complement':'互补关联编码','redundant':'双变量关联编码'}
    with plt.rc_context({'font.family':font,'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
        a=ax[0,0]
        a.barh([0,1],[0,1024],height=.48,color=['#B34D00','#0072B2'])
        a.set_yticks([0,1],['固定算符／字典序映射','期望值符号／不限线路'])
        a.set_xlim(0,1120);a.set_xlabel('可表示的唯一归属分配数（共 1024 种）')
        a.set_title('甲  7 比特的两种编码不是同一个模型',loc='left',fontweight='bold')
        a.text(15,0,'0',va='center');a.text(1040,1,'1024',va='center')
        a.text(.03,.94,'10 客户、2 车辆、同一组 20 个 Z 关联\n符号表示检验已知目标，不是求解成绩',transform=a.transAxes,va='top',fontsize=9)
        a.set_ylim(-.65,2)
        a=ax[0,1]
        categories=[(10,'classical'),(10,'complement'),(10,'redundant'),(16,'classical'),(16,'complement')]
        labels=[]
        for i,(n,model) in enumerate(categories):
            rows=[r for r in train['rows'] if r['n']==n and r['model']==model]
            values=np.array([r['route_gap_percent'] for r in rows])
            x=i+np.linspace(-.13,.13,len(values))
            a.scatter(x,values,c=colors[model],s=23,alpha=.65,marker={'classical':'s','complement':'o','redundant':'^'}[model])
            a.plot([i-.22,i+.22],[values.mean()]*2,color='black',lw=2)
            labels.append(f'{n} 客户\n'+names[model])
        a.set_xticks(range(len(labels)),labels,fontsize=8)
        a.set_ylim(bottom=0);a.set_ylabel('固定分配精确排序后，相对全局最优差距（%）')
        a.set_title('乙  可行分配尚未转化为路线质量优势',loc='left',fontweight='bold')
        a.text(.03,.97,'点：每次运行；横线：均值\n每模型 10 客户 12 次；16 客户 8 次',transform=a.transAxes,va='top',fontsize=9)
        a=ax[1,0]
        labels=list(dict.fromkeys(r['source_label'] for r in h1['rows']))
        y=np.arange(len(labels))
        feasible=[sum(r['result']['feasible'] for r in h1['rows'] if r['source_label']==s) for s in labels]
        exact=[sum(r['identical_assignment'] for r in h1['rows'] if r['source_label']==s) for s in labels]
        short=['10 客户·双变量·100','10 客户·双变量·101','10 客户·互补·100','10 客户·互补·101',
               '16 客户·互补·200','24 客户·互补·300','40 客户·互补·400','60 客户·互补·500']
        a.barh(y-.16,feasible,height=.3,color='#009E73',label='原始分配可行')
        a.barh(y+.16,exact,height=.3,color='#0072B2',hatch='//',label='完整恢复理想分配')
        a.set_yticks(y,short,fontsize=8);a.invert_yaxis();a.set_xticks([0,1,2,3]);a.set_xlim(0,3.4)
        a.set_xlabel('成功次数／3 次重复（每次 4096 次测量）')
        a.set_title('丙  第一轮白桦真机，未修复容量',loc='left',fontweight='bold')
        a.legend(loc='lower right',fontsize=8)
        a=ax[1,1]
        originals=list(dict.fromkeys(r['original_id'] for r in h2['rows']))
        for j,original in enumerate(originals):
            group=[r for r in h2['rows'] if r['original_id']==original]
            ys=[]
            for suffix in ('','_margin_l4','_margin_l8'):
                ys.append(sum(r['original_identical'] for r in group if r['source_label']==original+suffix))
            n=group[0]['n']
            a.plot(np.arange(3)+.04*(j-1),ys,marker=['o','s','^'][j],lw=1.5,label=f'{n} 客户')
        a.set_xticks([0,1,2],['原线路同批重跑','4 层间隔训练','8 层间隔训练'])
        a.set_yticks([0,1,2,3]);a.set_ylim(-.15,3.5);a.set_ylabel('恢复最初分配次数／3 次重复')
        a.set_title('丁  同批复验：保持分配的读出修正',loc='left',fontweight='bold')
        a.legend(loc='upper left',fontsize=9,ncol=3)
        fig.suptitle('PCE-CVRP：表示能力、优化质量与真机恢复必须分别验证',fontsize=16,fontweight='bold')
        fig.savefig(HERE/'结果总览.png',dpi=200,facecolor='white')
        fig.savefig(HERE/'结果总览.svg',facecolor='white')
        plt.close(fig)
    caption='本图为内部科研汇报，不声称符合特定期刊规范。甲：完整目标集合的确定性表示计数。乙：全部原始运行，点无统计置信区间；相对差距=100×(距离/精确最优值−1)。丙、丁：每线路三次硬件重复，三次不是独立问题实例，线条仅连接处理条件，不代表连续变化。失败和零成功均保留。所有硬件结果未做读出缓解或容量修复。来源：results、training/evaluation.json、hardware/analysis.json、hardware_refined/analysis.json。'
    (HERE/'图注与数据来源.txt').write_text(caption,encoding='utf-8')
    paths=['training/evaluation.json','hardware/analysis.json','hardware_refined/analysis.json','results/representation_summary.json']
    manifest=dict(purpose='内部科研汇报',journal_requirements='未指定',size_inches=[13,9],png_dpi=200,
                  aggregation='见图注；不筛除失败；不展示未经计算的置信区间',matplotlib=matplotlib.__version__,
                  source_sha256={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in paths})
    (HERE/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()

"""修正固定双车枚举的客户选择，旧结果完整归档。"""
from final_objective_accelerated import *
from evaluate_all import process
def main():
    records=[]
    for p in sorted((HERE/'classical').glob('*/*.json')):
        d=read(p);old=d['rows'][-1]
        if old.get('selection_version')=='within_two_vehicles':continue
        save(HERE/'classical_prior_selection'/p.relative_to(HERE/'classical'),d)
        S=setup(d['instance']);E=Evaluator(S,'NEW_DECODER',8 if S['n']==30 else 9)
        v=np.array(old['initial_assignment'],np.int64)
        vehicles=sorted(range(S['K']),key=lambda k:(-int(np.sum(v==k)),k))[:2]
        nodes=np.flatnonzero(np.isin(v,vehicles))[:8];assert len(nodes)==8
        best=(float('inf'),float('inf'));winner=None
        for bits in range(256):
            w=v.copy();w[nodes]=[vehicles[(bits>>i)&1] for i in range(8)];r=E.assess(w)
            if (r['overload'],r['distance'])<best:best=(r['overload'],r['distance']);winner=w
        d['rows'][-1]=process(E,winner,dict(method='FIXED_8_TWO_VEHICLES_256',candidate_budget=256,
            initial_assignment=v.tolist(),customers=nodes.tolist(),vehicles=vehicles,selection_version='within_two_vehicles',
            scope='固定选择客户最多的两车（同数按编号），取其中编号最小的8客户，枚举256种双车分配；其他客户不变。'))
        d['corrected_local_enumeration_ledger']=E.ledger();save(p,d)
        records.append(dict(instance=d['instance'],seed=d['seed'],vehicles=vehicles,customers=nodes.tolist()))
    save(HERE/'classical_control_correction.json',dict(records=records,reason='原固定前8客户可能来自两车之外，不能称为原两车内部局部枚举；保留旧结果并按原两车内客户重算。'))
    print('双车枚举修正完成',len(records),flush=True)
if __name__=='__main__':main()

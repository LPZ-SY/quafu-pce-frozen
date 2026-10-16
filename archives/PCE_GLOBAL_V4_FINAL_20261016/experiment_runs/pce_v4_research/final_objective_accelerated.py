"""用户更新目标：训练中直接评价同一确定性精修器的最终路线。"""
from core import *
from route_acceleration import accelerated_route_cost

@njit(cache=True)
def cost_key(key,D,n,cache):
    if key not in cache:
        nodes=np.empty(n,np.int64);j=0
        for i in range(n):
            if key & (1<<i):nodes[j]=i+1;j+=1
        cache[key]=accelerated_route_cost(D,nodes[:j])
    return cache[key]

@njit(cache=True)
def fast_refine(initial,D,q,cap,K,cache,budget=10000):
    n=len(initial);v=initial.copy();keys=np.zeros(K,np.int64);loads=np.zeros(K)
    for i in range(n):keys[v[i]]|=1<<i;loads[v[i]]+=q[i]
    costs=np.array([cost_key(keys[k],D,n,cache) for k in range(K)])
    over=np.maximum(0,loads-cap).sum();distance=costs.sum();evals=0;accepted=0
    repair=v.copy();has_repair=over==0;repair_evaluations=0 if has_repair else -1
    while evals<budget:
        best_over,best_dist=over,distance;mi,mj,mb=-1,-1,-1
        ca_best,cb_best=0.,0.
        for i in range(n):
            if evals>=budget:break
            a=v[i]
            for b in range(K):
                if evals>=budget:break
                if a==b:continue
                evals+=1;la,lb=loads[a]-q[i],loads[b]+q[i]
                no=over-max(0.,loads[a]-cap)-max(0.,loads[b]-cap)+max(0.,la-cap)+max(0.,lb-cap)
                if no>best_over:continue
                ca=cost_key(keys[a]^(1<<i),D,n,cache);cb=cost_key(keys[b]|(1<<i),D,n,cache)
                nd=distance-costs[a]-costs[b]+ca+cb
                if no<best_over or (no==best_over and nd<best_dist):
                    best_over,best_dist=no,nd;mi,mj,mb=i,-1,b;ca_best,cb_best=ca,cb
        for i in range(n):
            if evals>=budget:break
            for j in range(i+1,n):
                if evals>=budget:break
                a,b=v[i],v[j]
                if a==b:continue
                evals+=1;la,lb=loads[a]-q[i]+q[j],loads[b]-q[j]+q[i]
                no=over-max(0.,loads[a]-cap)-max(0.,loads[b]-cap)+max(0.,la-cap)+max(0.,lb-cap)
                if no>best_over:continue
                mask=(1<<i)|(1<<j)
                ca=cost_key(keys[a]^mask,D,n,cache);cb=cost_key(keys[b]^mask,D,n,cache)
                nd=distance-costs[a]-costs[b]+ca+cb
                if no<best_over or (no==best_over and nd<best_dist):
                    best_over,best_dist=no,nd;mi,mj,mb=i,j,b;ca_best,cb_best=ca,cb
        if mi<0:break
        a,b=v[mi],mb
        if mj<0:
            keys[a]^=1<<mi;keys[b]|=1<<mi;loads[a]-=q[mi];loads[b]+=q[mi];v[mi]=b
        else:
            mask=(1<<mi)|(1<<mj);keys[a]^=mask;keys[b]^=mask
            loads[a]+=q[mj]-q[mi];loads[b]+=q[mi]-q[mj];v[mi],v[mj]=b,a
        costs[a],costs[b]=ca_best,cb_best;over,distance=best_over,best_dist;accepted+=1
        if not has_repair and over==0:repair=v.copy();has_repair=True;repair_evaluations=evals
    return v,repair,has_repair,evals,accepted,repair_evaluations,over,distance

class FinalEvaluator(Evaluator):
    def __init__(self,S,method,nq,layers=4):
        super().__init__(S,method,nq,layers)
        self.refined={};self.refinement_calls=0;self.refinement_evaluations=0;self.refinement_seconds=0.;self.cached_refinements=0

    def hybrid(self,v):
        raw=self.assess(v);self.refinement_calls+=1;key=v.tobytes()
        if key not in self.refined:
            start=time.perf_counter();S=self.S
            result=fast_refine(v,S['inst']['D'],S['q'],S['cap'],S['K'],self.cache)
            self.refinement_seconds+=time.perf_counter()-start;self.refinement_evaluations+=int(result[3])
            self.refined[key]=result
        else:self.cached_refinements+=1
        final,repair,has_repair,evals,accepted,revals,over,distance=self.refined[key]
        changed=int(np.sum(final!=v))
        # 最终不可行时不使用路线长度排名；百万级失败惩罚严格隔离可行得分区间。
        main=distance/self.scale if over==0 else 1e6+30*over/self.S['q'].sum()
        score=main+.35*raw['overload']/self.S['q'].sum()+.10*changed/self.S['n']
        return float(score),dict(raw=raw,final=dict(feasible=bool(over==0),overload=float(over),distance=float(distance)),
            final_assignment=final.tolist(),repair_assignment=repair.tolist() if has_repair else None,
            changed_customers=changed,changed_demand=float(self.S['q'][final!=v].sum()),
            refinement_candidate_evaluations=int(evals),refinement_accepted_moves=int(accepted),repair_evaluations=int(revals))

    def objective(self,theta,rng,diagnostics=True):
        pp=probabilities(theta,self.nq,self.layers);self.states+=1;self.query_count+=1
        e=scores_from_moments(walsh(pp),self.fams,self.masks);ref=self.decode(e);h0,a0=self.hybrid(ref)
        ee,_=sample(pp,self.fams,self.masks,rng,batches=4);self.shots+=4*4096
        losses=[];aa=[];flips=[]
        for values in ee:
            v=self.decode(values);h,a=self.hybrid(v);losses.append(h);aa.append(a);flips.append(float(np.mean(v!=ref)))
        loss=.35*h0+.55*np.mean(losses)+.10*np.mean(flips)
        return float(loss),dict(reference_final_distance=a0['final']['distance'],
            reference_overload=a0['raw']['overload'],reference_changed=a0['changed_customers'],
            measured_final_distance=float(np.mean([a['final']['distance'] for a in aa])),
            measured_final_feasibility=float(np.mean([a['final']['feasible'] for a in aa])),
            measured_raw_overload=float(np.mean([a['raw']['overload'] for a in aa])),
            measured_changed=float(np.mean([a['changed_customers'] for a in aa])),customer_flip_rate=float(np.mean(flips)))

    def validate(self,theta,rng,batches=16,shots=4096):
        rows,counts,e,values=self.readouts(theta,rng,batches,shots);ref=self.decode(e);h0,a0=self.hybrid(ref);hybrid=[]
        for row,values_i in zip(rows,values):
            h,a=self.hybrid(np.array(row['assignment'],np.int64));hybrid.append(h);row.update(a)
        loss=.35*h0+.55*np.mean(hybrid)+.10*np.mean([r['customer_flip_rate'] for r in rows])
        result=summary(rows)
        result.update(final_feasibility_rate=float(np.mean([r['final']['feasible'] for r in rows])),
            feasible_final_distance=float(np.mean([r['final']['distance'] for r in rows if r['final']['feasible']])) if any(r['final']['feasible'] for r in rows) else None,
            mean_changed_customers=float(np.mean([r['changed_customers'] for r in rows])),hybrid_loss=float(loss))
        return result,rows,counts,e,values

    def ledger(self):
        d=super().ledger();d.update(refinement_calls=self.refinement_calls,refinement_cache_hits=self.cached_refinements,
            actual_refinement_candidate_evaluations=self.refinement_evaluations,refinement_seconds=self.refinement_seconds,
            refinement_budget_per_assignment=10000,unique_refined_assignments=len(self.refined))
        return d

def verify_fast():
    from evaluate_all import refine_stages
    details=[];rng=np.random.default_rng(66531)
    for name in ('B-n31-k5','A-n45-k7'):
        S=setup(name);E=FinalEvaluator(S,'FINAL_JOINT',8 if S['n']==30 else 9)
        vectors=[rng.integers(S['K'],size=S['n'],dtype=np.int64) for _ in range(6)]
        chosen=read(HERE/'PCE_GLOBAL_V4_FINAL.json')['implementation']
        phase='confirmation' if S['n']==30 else 'scale';sd=2001 if S['n']==30 else 3001
        vectors.append(np.array(read(HERE/'models'/phase/name/f'{chosen}_{sd}'/'result.json')['reference_assignment'],np.int64))
        for v in vectors:
            a,repair,meta=refine_stages(S,v);start=time.perf_counter();_,b=E.hybrid(v)
            assert np.array_equal(a,b['final_assignment'])
            assert meta['candidate_evaluations']==b['refinement_candidate_evaluations']
            assert (repair is None and b['repair_assignment'] is None) or np.array_equal(repair,b['repair_assignment'])
            details.append(dict(instance=name,candidate_evaluations=meta['candidate_evaluations'],seconds=time.perf_counter()-start))
    save(HERE/'final_refiner_verification.json',dict(passed=True,details=details,scope='14份正式规模分配与原Python精修器逐项一致；候选顺序、预算、结果与首次可行中间态一致。'))
    print('精修器等价核验通过',details,flush=True)

if __name__=='__main__':verify_fast()

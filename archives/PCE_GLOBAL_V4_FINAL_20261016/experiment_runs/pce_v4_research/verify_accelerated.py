"""完整精修逐项等价核验；正式实例，固定独立随机分配。"""
from final_objective import *
from final_objective_accelerated import fast_refine as accelerated
from route_acceleration import verify
def main():
    verify();rng=np.random.default_rng(887281);details=[]
    for name in ('B-n31-k5','A-n45-k7'):
        S=setup(name)
        for j in range(20):
            v=rng.integers(S['K'],size=S['n'],dtype=np.int64)
            args=(v,S['inst']['D'],S['q'],S['cap'],S['K'])
            a=fast_refine(*args,Dict.empty(types.int64,types.float64))
            b=accelerated(*args,Dict.empty(types.int64,types.float64))
            assert all(np.array_equal(x,y) for x,y in zip(a,b)),(name,j)
            details.append(dict(instance=name,index=j,candidates=int(a[3]),distance=float(a[-1])))
    save(HERE/'complete_refiner_acceleration_verification.json',dict(passed=True,details=details,
         original_sha256=digest(HERE/'final_objective.py'),accelerated_sha256=digest(HERE/'final_objective_accelerated.py'),
         scope='40份正式规模分配：最终分配、首次可行分配、候选评价数、接受步数、最终距离及超载逐项一致。'))
    print('完整精修等价核验通过',len(details),flush=True)
if __name__=='__main__':main()

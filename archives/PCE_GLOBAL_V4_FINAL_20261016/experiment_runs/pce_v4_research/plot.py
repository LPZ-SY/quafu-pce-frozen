"""显示训练种子差异，禁止将测量重复当作独立算法重复。"""
from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from report import LABELS,all_data

def main():
    out=HERE/'figures';out.mkdir(exist_ok=True)
    font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10})
    colors={'FINAL_JOINT':'#0072B2','FINAL_RANDOM_SEARCH':'#D55E00'}
    fields=[('training_best_loss','截至当前最优训练损失'),('measured_final_distance','训练读出经精修后的路线'),('measured_changed','训练读出净跨车客户数')]
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,(phase,title) in enumerate([('final_confirmation','30客户／5车'),('final_scale','44客户／7车')]):
        for method in colors:
            models=[read(p) for p in (HERE/'models'/phase).glob(f'*/{method}_*/result.json')]
            if not models:continue
            for col,(field,label) in enumerate(fields):
                ax=axes[row,col];values=[]
                for m in models:
                    x=np.array([r['query'] for r in m['trace']]);y=np.array([r[field] if field!='measured_final_distance' or r.get('measured_final_feasibility')==1 else np.nan for r in m['trace']]);values.append(y)
                    ax.plot(x,y,color=colors[method],alpha=.13,linewidth=.6)
                ys=np.array(values);median=np.nanmedian(ys,axis=0);lo,hi=np.nanquantile(ys,[.25,.75],axis=0)
                ax.fill_between(x,lo,hi,color=colors[method],alpha=.12)
                ax.plot(x,median,color=colors[method],linestyle='-' if method=='FINAL_JOINT' else '--',linewidth=1.8,label=f'{LABELS[method]}（{len(models)}种子）')
                ax.set(xlabel='参数查询次数',ylabel=label,title=f'{title} · {label}');ax.spines[['top','right']].set_visible(False)
                if col==0 and np.max(ys)>100:ax.set_yscale('log');ax.set_ylabel(label+'（对数刻度）')
        axes[row,0].legend(fontsize=8)
    fig.suptitle('训练轨迹：细线为各种子，粗线为中位数，阴影为种子间四分位范围',fontsize=12)
    fig.savefig(out/'training_curves.png',dpi=200,facecolor='white');fig.savefig(out/'training_curves.pdf',facecolor='white');plt.close(fig)
    data,groups=all_data();methods=['LEGACY','NEW_DECODER','NEW_JOINT','V42_GEOMETRY','FINAL_JOINT','FINAL_RANDOM_SEARCH','RANDOM_THETA']
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');rng=np.random.default_rng(20261015)
    for row,(cohort,name,title) in enumerate([('confirmation','B-n31-k5','30客户／5车'),('scale','A-n45-k7','44客户／7车')]):
        for col,(field,label) in enumerate([('feasible_final_distance','最终可行路线长度'),('raw_mean_overload','精修前平均超载'),('mean_changed_customers','平均净跨车客户数')]):
            ax=axes[row,col]
            for i,method in enumerate(methods):
                vals=np.array([m['summary']['4096'][field] for m in groups.get((cohort,name,method),[]) if m['summary']['4096'][field] is not None])
                if not len(vals):continue
                ax.scatter(i+rng.uniform(-.16,.16,len(vals)),vals,s=13,alpha=.5,color='#0072B2' if method=='FINAL_JOINT' else '#777777')
                means=vals[rng.integers(len(vals),size=(20000,len(vals)))].mean(axis=1);lo,hi=np.quantile(means,[.025,.975])
                ax.errorbar(i,float(vals.mean()),yerr=[[float(vals.mean()-lo)],[float(hi-vals.mean())]],fmt='D',color='black',markersize=4,capsize=4)
            ax.set(xticks=range(len(methods)),xticklabels=['旧版','新解码','原始联合','几何改造','最终联合','随机搜索','随机参数'],ylabel=label,title=title)
            ax.tick_params(axis='x',labelrotation=35);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('独立测试：每点为一个训练种子的64批读出均值；黑点与误差线为种子均值及95%重抽样区间',fontsize=11)
    fig.savefig(out/'final_comparison.png',dpi=200,facecolor='white');fig.savefig(out/'final_comparison.pdf',facecolor='white');plt.close(fig)
    save(out/'figure_manifest.json',dict(audience='科研汇报与内部研究报告；未声明满足特定期刊要求',
        source=['models/*/*/*/result.json','final_tests/*/*/*.json'],training='无平滑；展示全部已完成种子的记录、中位数及25%—75%分位区间；若同一记录含最终失败读出，其距离留空',
        comparison='先汇总每模型64次读出，再以种子为单位20000次重抽样；路线只统计最终可行读出；未完成组不补零',
        seed=20261015,files=[dict(file=p.name,sha256=digest(p)) for p in out.glob('*.png')]))
    print('曲线与结果图已生成',flush=True)

if __name__=='__main__':main()

"""Reproduces Figures 1-6 of the paper from the CSV files in results/."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.path.join(HERE,'results')+'/'
OUTDIR=os.path.join(HERE,'figures')+'/'
os.makedirs(OUTDIR,exist_ok=True)

def ablation_summary():
    p=pd.read_csv(D+'ablation__ablacion_path.csv'); r=pd.read_csv(D+'ablation__ablacion.csv')
    rows=[]
    for (h,ds),g in p.groupby(['config_hash','data_seed']):
        best=g.groupby('k').W.min()
        for (s_,i),gi in g.groupby(['seed','init']):
            w=gi.set_index('k').W.sort_index(); e=(w-best)/best*100
            rows.append(dict(config_hash=h,data_seed=ds,seed=s_,init=i,exc=e.mean()))
    t=r[r.criterion=='ch'][['config_hash','data_seed','seed','init','path_time_s']]
    return pd.DataFrame(rows).merge(t,on=['config_hash','data_seed','seed','init'])
CW,CK,CI,CM='#257de0','#f47230','#00b894','#6a4ce2'
TXT,SEC,GRID='#191919','#666666','#e6e6e6'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.edgecolor':SEC,'axes.labelcolor':TXT,'xtick.color':SEC,'ytick.color':SEC,
 'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.dpi':300,'axes.titlesize':9.5,'axes.titleweight':'bold'})
def save(f,name): f.tight_layout(); f.savefig(OUTDIR+name,bbox_inches='tight',facecolor='white'); plt.close(f)

# ---- Fig 1: Covertype excess inertia per k, 3 seeds
p=pd.concat([pd.read_csv(D+'real_seed0__reales_path.csv'),pd.read_csv(D+'real_covertype_seeds1-2__reales_path.csv')])
p=p[p.dataset=='covertype']; best=p.groupby('k').W.min()
f,ax=plt.subplots(1,2,figsize=(7.2,2.9))
for s,ls in zip([0,1,2],['-','--',':']):
    g=p[(p.init=='kmeansll')&(p.seed==s)].set_index('k').W.sort_index()
    ax[0].plot(g.index,(g-best)/best*100,ls,color=CK,lw=1.4,marker='o',ms=3,label=f'k-means|| (seed {s})')
g=p[(p.init=='warm')&(p.seed==0)].set_index('k').W.sort_index()
ax[0].plot(g.index,(g-best)/best*100,'-',color=CW,lw=2,marker='s',ms=3.5,label='Proposed (all seeds)')
ax[0].set_xlabel('Number of clusters k'); ax[0].set_ylabel('Excess inertia (%)'); ax[0].set_title('(a) Excess inertia over best known W(k)')
ax[0].legend(fontsize=7,loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.0)); ax[0].set_ylim(-0.3,7.6); ax[0].set_xticks(range(2,21,2))
for s,ls in zip([0,1,2],['-','--',':']):
    g=p[(p.init=='kmeansll')&(p.seed==s)].set_index('k').W.sort_index(); dw=g.diff()/g.shift()*100
    ax[1].plot(dw.index,dw,ls,color=CK,lw=1.4,marker='o',ms=3)
g=p[(p.init=='warm')&(p.seed==0)].set_index('k').W.sort_index(); dw=g.diff()/g.shift()*100
ax[1].plot(dw.index,dw,'-',color=CW,lw=2,marker='s',ms=3.5)
ax[1].axhline(0,color=TXT,lw=0.8); ax[1].set_xlabel('Number of clusters k'); ax[1].set_ylabel('Relative change W(k) vs W(k−1) (%)')
ax[1].set_title('(b) Step-to-step change (> 0 = violation)'); ax[1].set_xticks(range(2,21,2))
viol=[]
for s in [0,1,2]:
    g=p[(p.init=='kmeansll')&(p.seed==s)].set_index('k').W.sort_index()
    for k in g.index[1:]:
        if g[k]>g[k-1]: viol.append((k,(g[k]-g[k-1])/g[k-1]*100))
ax[1].scatter([k for k,_ in viol],[v for _,v in viol],s=90,facecolors='none',edgecolors=TXT,linewidths=1.0,zorder=5,label='Monotonicity violation')
ax[1].legend(fontsize=7.5,loc='lower right')
save(f,'fig1_covertype_trajectory.png')

# ---- Fig 2: mechanism — Lloyd iterations and cumulative time per k (12 synthetic scenarios)
p=pd.read_csv(D+'quality__calidad_path.csv'); p=p[p.k>=2]
f,ax=plt.subplots(1,2,figsize=(7.2,2.9))
for init,col,lab in [('kmeansll',CK,'k-means|| (restarted)'),('warm',CW,'Proposed')]:
    g=p[p.init==init].groupby('k')
    m=g.iters.mean(); q1=g.iters.quantile(0.25); q3=g.iters.quantile(0.75)
    ax[0].plot(m.index,m,color=col,lw=2,marker='o',ms=3,label=lab); ax[0].fill_between(m.index,q1,q3,color=col,alpha=0.15,lw=0)
    ct=p[p.init==init].sort_values('k').groupby('config_hash').time_s.cumsum(); p.loc[ct.index,'ct']=ct
    cm=p[p.init==init].groupby('k').ct.mean()
    ax[1].plot(cm.index,cm,color=col,lw=2,marker='o',ms=3,label=lab)
ax[0].set_xlabel('Number of clusters k'); ax[0].set_ylabel('Lloyd iterations at step k'); ax[0].set_title('(a) Lloyd iterations (mean, IQR band)')
ax[0].legend(fontsize=7.5); ax[0].set_xticks(range(2,21,2))
ax[1].set_xlabel('Number of clusters k'); ax[1].set_ylabel('Cumulative time (s)'); ax[1].set_title('(b) Cumulative trajectory time'); ax[1].set_xticks(range(2,21,2))
save(f,'fig2_mechanism.png')

# ---- Fig 3: ablation time vs quality
a=ablation_summary()
f,ax=plt.subplots(figsize=(4.6,3.3))
for init,col,lab,mk in [('kmeansll',CK,'k-means|| (restarted)','o'),('incpp',CI,'Incremental k-means++','^'),('warm',CW,'Proposed','s')]:
    g=a[a.init==init]; ax.scatter(g.path_time_s,g.exc,s=26,color=col,marker=mk,label=lab,edgecolor='white',linewidth=0.6,zorder=3)
ax.set_yscale('log'); ax.set_xlabel('Trajectory time, k = 2…20 (s)'); ax.set_ylabel('Mean excess inertia (%) — log scale')
ax.legend(fontsize=7.5,loc='upper left'); ax.set_title('Ablation: 3 data sets × 2 separations × 3 seeds')
save(f,'fig3_ablation.png')

# ---- Fig 4: stopping rule sensitivity
s=pd.read_csv(D+'stopping_rule__sensibilidad.csv'); s=s[s.criterion=='ch']
rules=[('conv_sklearn','scikit-learn\n(default)'),('conv_both','Combined\n(shift or ΔW)'),('conv_shift','MLlib\n(max shift)')]
f,ax=plt.subplots(1,2,figsize=(7.2,2.9)); x=np.arange(3); w=0.36
for j,(init,col,lab) in enumerate([('kmeansll',CK,'k-means|| (restarted)'),('warm',CW,'Proposed')]):
    m=[s[(s.tag==t)&(s.init==init)].path_time_s.mean() for t,_ in rules]
    ax[0].bar(x+(j-0.5)*w,m,w*0.92,color=col,label=lab,zorder=3)
    for i,(t,_) in enumerate(rules):
        v=s[(s.tag==t)&(s.init==init)].path_time_s.values; ax[0].scatter(np.full(3,x[i]+(j-0.5)*w),v,s=8,color=TXT,zorder=4)
ax[0].set_xticks(x,[l for _,l in rules],fontsize=8); ax[0].set_ylabel('Trajectory time, k = 2…25 (s)'); ax[0].legend(fontsize=7.5,loc='upper center',bbox_to_anchor=(0.5,1.0)); ax[0].set_ylim(0,620); ax[0].set_title('(a) Time by stopping rule (n = 10⁶)')
sp=[]
for t,_ in rules:
    pv=s[s.tag==t].pivot_table(index='data_seed',columns='init',values='path_time_s'); sp.append(pv.kmeansll/pv.warm)
ax[1].bar(x,[v.mean() for v in sp],0.5,color=CW,zorder=3)
for i,v in enumerate(sp):
    ax[1].scatter(np.full(len(v),x[i]),v,s=10,color=TXT,zorder=4); ax[1].text(x[i]+0.3,v.mean()+0.05,f'{v.mean():.2f}×',ha='center',color=TXT,fontsize=8.5)
ax[1].axhline(1,color=TXT,lw=0.8); ax[1].set_xticks(x,[l for _,l in rules],fontsize=8); ax[1].set_ylabel('Speedup (k-means|| / proposed)'); ax[1].set_title('(b) Speedup')
save(f,'fig4_stopping_rule.png')

# ---- Fig 5: MLlib comparison
m=pd.read_csv(D+'mllib__mllib.csv'); o=pd.read_csv(D+'mllib__mllib_mllib_once.csv'); mp=pd.read_csv(D+'mllib__mllib_path.csv')
f,ax=plt.subplots(1,2,figsize=(7.2,2.9))
items=[('warm','Proposed\n(k = 2…25)',CW),('kmeansll','k-means||\n(k = 2…25)',CK),('mllib_kmeans','MLlib KMeans\n(k = 2…25)',CM)]
vals=[m[m.init==i].path_time_s.values for i,_,_ in items]+[o[o.method=='mllib_bisecting_kmax'].fit_time_s.values]
labs=[l for _,l,_ in items]+['MLlib Bisecting\n(single fit, k = 25)']; cols=[c for _,_,c in items]+['#9c9c9c']
y=np.arange(4)[::-1]
ax[0].barh(y,[v.mean() for v in vals],0.6,color=cols,zorder=3)
for yi,v in zip(y,vals): ax[0].scatter(v,np.full(len(v),yi),s=9,color=TXT,zorder=4); ax[0].text(v.max()+25,yi,f'{v.mean():.0f} s',va='center',fontsize=8,color=TXT)
ax[0].set_yticks(y,labs,fontsize=7.5); ax[0].set_xlabel('Time (s)'); ax[0].set_title('(a) Computing time (n = 10⁶, 3 data sets)'); ax[0].set_xlim(0,1350)
rows=[]
for ds in [101,102,103]:
    wv=mp[(mp.data_seed==ds)&(mp.k==25)].set_index('init').W
    bo=o[(o.data_seed==ds)]; bis=bo[bo.method=='mllib_bisecting_kmax'].W_voronoi.iloc[0]; mk=bo[bo.method=='mllib_kmeans_kmax'].W_voronoi.iloc[0]
    ref=wv['warm']; rows.append(dict(kll=(wv['kmeansll']/ref-1)*100,mlpath=(wv['mllib_kmeans']/ref-1)*100,mk=(mk/ref-1)*100,bis=(bis/ref-1)*100))
r=pd.DataFrame(rows); names=[('kll','k-means||\n(path)',CK),('mlpath','MLlib KMeans\n(path)',CM),('mk','MLlib KMeans\n(single fit)',CM),('bis','MLlib Bisecting\n(single fit)','#9c9c9c')]
y=np.arange(4)[::-1]
ax[1].barh(y,[r[c].mean() for c,_,_ in names],0.6,color=[c for _,_,c in names],zorder=3)
for yi,(c,_,_) in zip(y,names): ax[1].scatter(r[c],np.full(3,yi),s=9,color=TXT,zorder=4); ax[1].text(max(r[c].max(),0)+0.05,yi,f'+{r[c].mean():.2f}%',va='center',fontsize=8,color=TXT)
ax[1].set_yticks(y,[l for _,l,_ in names],fontsize=7.5); ax[1].set_xlabel('Inertia at k = 25 relative to proposed (%)'); ax[1].set_title('(b) Quality at k = 25'); ax[1].set_xlim(0,2.1)
save(f,'fig5_mllib.png'); pass

# ---- Fig 6: real datasets
R=pd.concat([pd.read_csv(D+'real_seed0__reales.csv'),pd.read_csv(D+'real_covertype_seeds1-2__reales.csv')]); R=R[R.criterion=='ch']
dsn=[('covertype','Covertype\n581,012 × 54'),('susy_40pct','SUSY (40%)\n1,999,130 × 18'),('higgs_10pct','HIGGS (10%)\n1,100,106 × 28')]
f,ax=plt.subplots(figsize=(5.2,3.0)); x=np.arange(3); w=0.36
for j,(init,col,lab) in enumerate([('kmeansll',CK,'k-means|| (restarted)'),('warm',CW,'Proposed')]):
    for i,(d,_) in enumerate(dsn):
        v=R[(R.dataset==d)&(R.init==init)].path_time_s.values
        ax.bar(x[i]+(j-0.5)*w,v.mean(),w*0.92,color=col,zorder=3,label=lab if i==0 else None); ax.scatter(np.full(len(v),x[i]+(j-0.5)*w),v,s=9,color=TXT,zorder=4)
for i,(d,_) in enumerate(dsn):
    a_=R[(R.dataset==d)&(R.init=='kmeansll')].path_time_s.mean(); b_=R[(R.dataset==d)&(R.init=='warm')].path_time_s.mean()
    ax.text(x[i],R[(R.dataset==d)&(R.init=='kmeansll')].path_time_s.max()+30,f'{a_/b_:.2f}×',ha='center',fontsize=8.5,color=TXT)
ax.set_xticks(x,[l for _,l in dsn],fontsize=8); ax.set_ylabel('Trajectory time, k = 2…20 (s)'); ax.legend(fontsize=7.5,loc='upper left'); ax.set_ylim(0,880)
ax.set_title('Real data sets (Covertype: 3 seeds; SUSY, HIGGS: seed 0)')
save(f,'fig6_real.png')
print('ok')

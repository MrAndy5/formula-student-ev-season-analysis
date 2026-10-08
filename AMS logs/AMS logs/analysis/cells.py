import glob, os, numpy as np, pandas as pd
D="/Users/alvarogonzaleztabernero/dev/AMS logs"
logs={int(os.path.basename(f)[3:7]):pd.read_csv(f) for f in sorted(glob.glob(f"{D}/LOG*.CSV"))}
cc=[f"c{m}_{n}" for m in range(5) for n in range(19)]
# rest rows: |I_filt|<2A, not in Run
rest=pd.concat([d[(d.I_filt_mA.abs()<2000)&(d.fsm!=3)].assign(log=i) for i,d in logs.items()])
V=rest[cc].values.astype(float)
dev=V-np.median(V,1,keepdims=True)
print("rest rows",len(rest))
md=pd.Series(np.median(dev,0),index=cc)
print("rest deviation from pack median (mV), top/bottom:")
print(md.sort_values().head(8).round(0).to_dict()); print(md.sort_values().tail(8).round(0).to_dict())
print("per-module rest mean (mV):",[round(V[:,m*19:(m+1)*19].mean()) for m in range(5)])
# noise: median abs successive diff within each log at rest
nz={}
for i,d in logs.items():
    r=d[(d.I_filt_mA.abs()<2000)&(d.fsm!=3)][cc]
    if len(r)>20: nz[i]=r.diff().abs().mean()
nz=pd.DataFrame(nz).T
print("mean |dV| sample-to-sample at rest, noisiest cells:"); print(nz.mean().sort_values().tail(10).round(1).to_dict())
print("typical:",nz.mean().median().round(2))
# spikes: single-sample dips >20 mV that recover
sp=pd.Series(0,index=cc)
for i,d in logs.items():
    r=d[(d.I_filt_mA.abs()<2000)&(d.fsm!=3)][cc].values.astype(float)
    if len(r)<3: continue
    mid=r[1:-1]-(r[:-2]+r[2:])/2
    sp+=(np.abs(mid)>15).sum(0)
print("single-sample spikes >15mV at rest:",sp[sp>0].sort_values(ascending=False).head(12).to_dict())
# internal resistance under load
ld=pd.concat([d[(d.fsm==3)].assign(log=i) for i,d in logs.items()])
ld=ld[ld.mod_mask==31]
I=ld.I_filt_mA.values/1000
Vl=ld[cc].values.astype(float)
R={}
for j,c in enumerate(cc):
    A=np.vstack([I,np.ones_like(I)]).T
    # per-log intercept: regress on I with log dummies
    X=np.column_stack([I]+[(ld.log.values==k).astype(float) for k in ld.log.unique()])
    b=np.linalg.lstsq(X,Vl[:,j],rcond=None)[0]
    R[c]=-b[0]
R=pd.Series(R)
print("cell R from I_filt regression (mOhm): median",R.median().round(2),"min/max",R.min().round(2),R.max().round(2))
print("highest:",R.sort_values().tail(10).round(2).to_dict())
print("lowest:",R.sort_values().head(6).round(2).to_dict())
print("per-module median R:",[round(R[[f'c{m}_{n}' for n in range(19)]].median(),2) for m in range(5)])
Rp=np.linalg.lstsq(np.column_stack([I]+[(ld.log.values==k).astype(float) for k in ld.log.unique()]),ld.pack_mV.values/1000,rcond=None)[0][0]
print("pack R (Ohm) from pack_mV vs I_filt:",-Rp)

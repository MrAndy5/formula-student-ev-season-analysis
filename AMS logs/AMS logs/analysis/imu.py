import pandas as pd, numpy as np, sys
D="/Users/alvarogonzaleztabernero/dev/AMS logs"
for i in (74,93,94,97,100,101,105,106):
    m=pd.read_csv(f"{D}/IMU{i:04d}.CSV"); d=pd.read_csv(f"{D}/LOG{i:04d}.CSV")
    t=m.tick_ms.values/1000
    a=m[["ax_g","ay_g","az_g"]].values; g=m[["gx_rad_s","gy_rad_s","gz_rad_s"]].values
    # high-pass jerk: sample diff magnitude
    da=np.linalg.norm(np.diff(a,axis=0),axis=1)
    # 1 s windows
    w=100; n=len(a)//w
    stdw=np.array([a[k*w:(k+1)*w].std(0).max() for k in range(n)])
    gw=np.array([np.abs(g[k*w:(k+1)*w]).max() for k in range(n)])
    mean_a=a[:500].mean(0)
    print(f"IMU{i:04d}: static gravity vec {np.round(mean_a,3)} |g|={np.linalg.norm(mean_a):.3f}")
    moving=[k for k in range(n) if stdw[k]>0.01]
    if moving:
        # group contiguous
        seg=[];s=moving[0];p=s
        for k in moving[1:]:
            if k>p+2: seg.append((s,p)); s=k
            p=k
        seg.append((s,p))
        print("  active windows (acc std>0.01g):", ", ".join(f"{t[a0*w]:.0f}-{t[min((b0+1)*w,len(t)-1)]:.0f}s" for a0,b0 in seg))
    # spikes: jerk >0.05g in one 10ms sample
    idx=np.where(da>0.04)[0]
    # cluster spikes within 0.2s
    cl=[]
    for k in idx:
        if not cl or t[k]-cl[-1][-1]>0.2: cl.append([t[k],t[k]])
        else: cl[-1][-1]=t[k]
    # FSM changes
    fs=d.fsm.values; ch=[(d.tick_ms.iloc[k]/1000,fs[k-1],fs[k]) for k in range(1,len(fs)) if fs[k]!=fs[k-1]]
    ts=d.tsms.values; tch=[(d.tick_ms.iloc[k]/1000,ts[k]) for k in range(1,len(ts)) if ts[k]!=ts[k-1]]
    print("  FSM changes:",[(round(a,2),f"{b}->{c}") for a,b,c in ch], " TSMS:",[(round(a,2),b) for a,b in tch])
    print(f"  IMU step spikes (|da|>0.04g) clusters n={len(cl)}:", ", ".join(f"{c0:.2f}" for c0,c1 in cl[:40]))

import glob, os, numpy as np, pandas as pd
D="/Users/alvarogonzaleztabernero/dev/AMS logs"
for f in sorted(glob.glob(f"{D}/LOG*.CSV")):
    i=int(os.path.basename(f)[3:7]); d=pd.read_csv(f)
    fs=d.fsm.values
    for k in np.where((fs[1:]==1)&(fs[:-1]!=1))[0]+1:
        j=k
        while j<len(d) and d.fsm.iloc[j]==1: j+=1
        seg=d.iloc[k-1:j+1]
        pack=d.pack_mV.iloc[k]/1000
        dc=seg.dcbus_V.values
        out=d.fsm.iloc[j] if j<len(d) else None
        print(f"LOG{i:04d} t={d.tick_ms.iloc[k]/1000:6.2f}s pack={pack:.1f}V -> {'Run' if out in (2,3) else 'ERR' if out==5 else out} dur={(d.tick_ms.iloc[min(j,len(d)-1)]-d.tick_ms.iloc[k])/1000:.2f}s  dc: {' '.join(str(x) for x in dc)}  final%={dc[-1]/pack*100:.0f} Iraw: {' '.join(f'{x/1000:.1f}' for x in seg.I_raw_mA.values[:8])}")

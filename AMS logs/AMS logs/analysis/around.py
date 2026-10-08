import sys, pandas as pd, numpy as np
D="/Users/alvarogonzaleztabernero/dev/AMS logs"
i=int(sys.argv[1]); t0=float(sys.argv[2])*1000; w=float(sys.argv[3])*1000 if len(sys.argv)>3 else 2000
d=pd.read_csv(f"{D}/LOG{i:04d}.CSV")
tc=[c for c in d.columns if c[0]=='t' and c[1].isdigit()]
sel=d[(d.tick_ms>=t0-w)&(d.tick_ms<=t0+w)]
for _,r in sel.iterrows():
    s=[c for c in tc if r[c]==-32768]
    print(f"{r.tick_ms:7d} fsm={r.fsm} mode={r['mode']} ok={r.ams_ok} f={r.fault}/{r.detail} tsms={r.tsms} dash={r.dash_chg} mm={r.mod_mask} pack={r.pack_mV/1000:.1f} I={r.I_filt_mA/1000:.1f}/{r.I_raw_mA/1000:.1f} dc={r.dcbus_V} vmin={r.vmin_mV} vmax={r.vmax_mV} tmin={r.tmin_C} tmax={r.tmax_C} open={s[:8]}{'...+'+str(len(s)-8) if len(s)>8 else ''}")

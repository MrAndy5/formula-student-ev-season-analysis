import glob, os, sys
import numpy as np, pandas as pd

D = "/Users/alvarogonzaleztabernero/dev/AMS logs"
S = -32768
FSM = {0: "Start", 1: "Pre", 2: "Trans", 3: "Run", 4: "Chg", 5: "ERR"}
FAULT = {0: "None", 2: "ModOffline", 3: "BmsStale", 4: "UV", 5: "OV", 8: "IsensFault", 9: "IStale",
         10: "OverI", 11: "VcuStale", 12: "FsmError", 13: "NtcDisc", 14: "ChgStale", 15: "ChgTsms", 16: "OpenWire"}


def load(i):
    d = pd.read_csv(f"{D}/LOG{i:04d}.CSV")
    return d


def runs(s, t):
    out = []
    prev = None
    for v, tt in zip(s, t):
        if v != prev:
            out.append((tt, v))
            prev = v
    return out


def mask(x):
    return "".join(f"M{m}" for m in range(5) if x >> m & 1) or "-"


for f in sorted(glob.glob(f"{D}/LOG*.CSV")):
    i = int(os.path.basename(f)[3:7])
    d = load(i)
    if d.empty:
        print(f"LOG{i:04d}: header only"); continue
    t = d.tick_ms.values
    dt = np.diff(t)
    tcols = [c for c in d.columns if c.startswith("t") and "_" in c and c[1].isdigit()]
    ccols = [c for c in d.columns if c.startswith("c") and c[1].isdigit()]
    T = d[tcols].values
    fsm_runs = runs(d.fsm.values, t)
    fsm_str = " ".join(f"{FSM[v]}@{tt/1000:.1f}" for tt, v in fsm_runs)
    last = d.iloc[-1]
    flt = d[d.fault != 0]
    ftxt = "-"
    if len(flt):
        r0 = flt.iloc[0]
        det = mask(r0.detail) if r0.fault in (2, 13, 16) else r0.detail
        ftxt = f"{FAULT.get(r0.fault, r0.fault)}({det})@{r0.tick_ms/1000:.2f}s"
    # NTCs that are sentinel in some rows but valid in others
    sent = (T == S)
    s1 = sent[1:]
    frac = s1.mean(0)
    persist = [f"{tcols[j]}({frac[j]*100:.0f}%)" for j in range(T.shape[1]) if frac[j] > 0.2]
    pj = [j for j in range(T.shape[1]) if frac[j] > 0.2]
    s2 = s1.copy(); s2[:, pj] = False
    ev = []
    for k in np.where(s2.any(1))[0]:
        r = d.iloc[k+1]
        ch = [tcols[j] for j in np.where(s2[k])[0]]
        ev.append(f"{r.tick_ms/1000:.2f}s[{FSM[r.fsm]}]:{','.join(ch[:6])}{'..+'+str(len(ch)-6) if len(ch)>6 else ''}")
    Tv = np.where(sent, np.nan, T)
    I = d.I_filt_mA.values / 1000
    print(f"LOG{i:04d}: {len(d)} rows {t[0]/1000:.1f}-{t[-1]/1000:.1f}s gaps>300ms={int((dt>300).sum())} maxgap={dt.max() if len(dt) else 0}ms")
    print(f"   FSM: {fsm_str}")
    print(f"   modes={sorted(set(d['mode']))} tsms={sorted(set(d.tsms))} dash_presses={int((np.diff(d.dash_chg.values)==1).sum())} mod_mask={sorted(set(d.mod_mask))} fault={ftxt}")
    print(f"   pack {d.pack_mV.min()/1000:.1f}-{d.pack_mV.max()/1000:.1f}V  I_filt {I.min():.1f}..{I.max():.1f}A  I_raw {d.I_raw_mA.min()/1000:.1f}..{d.I_raw_mA.max()/1000:.1f}A  Idcdc {d.Idcdc_mA.min()/1000:.1f}..{d.Idcdc_mA.max()/1000:.1f}A  dcbus {d.dcbus_V.min()}-{d.dcbus_V.max()}V")
    print(f"   vmin {d.vmin_mV.min()}-{d.vmin_mV.max()} vmax {d.vmax_mV.min()}-{d.vmax_mV.max()}  T {np.nanmin(Tv):.0f}..{np.nanmax(Tv):.0f}C  tmax_col {d.tmax_C[d.tmax_C<1000].max()}")
    print(f"   NTC mostly-open: {persist}  transient-open rows={len(ev)}: {' | '.join(ev[:8])}{' ...' if len(ev)>8 else ''}")
    imu = f"{D}/IMU{i:04d}.CSV"
    if os.path.exists(imu):
        m = pd.read_csv(imu)
        mt = m.tick_ms.values
        mdt = np.diff(mt)
        a = np.sqrt(m.ax_g**2 + m.ay_g**2 + m.az_g**2)
        g = np.sqrt(m.gx_rad_s**2 + m.gy_rad_s**2 + m.gz_rad_s**2)
        print(f"   IMU: {len(m)} rows {mt[0]/1000:.1f}-{mt[-1]/1000:.1f}s gaps>20ms={int((mdt>20).sum())} maxgap={mdt.max()}ms  |a| {a.min():.2f}..{a.max():.2f}g std={a.std():.3f}  |w| max {g.max():.2f}rad/s  ax[{m.ax_g.min():.2f},{m.ax_g.max():.2f}] ay[{m.ay_g.min():.2f},{m.ay_g.max():.2f}] az[{m.az_g.min():.2f},{m.az_g.max():.2f}]")

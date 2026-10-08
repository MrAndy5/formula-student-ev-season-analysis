import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/alvarogonzaleztabernero/dev/AMS logs"
OUT = "/Users/alvarogonzaleztabernero/dev/AMS logs/analysis"
BLUE, ORANGE, AQUA, YELLOW, MAG = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CRIT, INK, INK2, GRID, SURF = "#d03b3b", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "legend.frameon": False, "lines.linewidth": 1.6})
L = lambda i: pd.read_csv(f"{D}/LOG{i:04d}.CSV")
I_ = lambda i: pd.read_csv(f"{D}/IMU{i:04d}.CSV")

fig, ax = plt.subplots(3, 2, figsize=(13, 12.5))
fig.subplots_adjust(hspace=0.42, wspace=0.22, top=0.94, bottom=0.05, left=0.06, right=0.98)
fig.suptitle("AMS logs 0074–0106: fault diagnosis", fontsize=13, fontweight="bold", x=0.06, ha="left")

# A precharge
a = ax[0, 0]
pre = [(74, 5.5), (75, 27.25), (77, 7.0), (86, 14.5), (88, 7.5), (89, 11.5), (91, 49.5), (93, 16.25), (95, 14.75),
       (96, 8.5), (97, 8.5), (98, 15.5), (99, 17.5), (102, 10.5), (102, 120.0), (102, 155.5), (105, 10.5), (106, 12.0)]
for n, (i, t0) in enumerate(pre):
    d = L(i); s = d[(d.tick_ms / 1000 >= t0 - 0.25) & (d.tick_ms / 1000 <= t0 + 5.25)]
    pack = d.pack_mV[d.tick_ms / 1000 >= t0].iloc[0] / 1000
    fail = s.fsm.isin([3, 2]).sum() == 0
    s = s[s.fsm.isin([0, 1, 2, 3])] if not fail else s[s.fsm != 5]
    a.plot(s.tick_ms / 1000 - t0, 100 * s.dcbus_V / pack, color=CRIT if fail else BLUE, alpha=0.8, lw=1.4)
a.axhline(95, color=INK2, ls="--", lw=1); a.text(3.9, 96.5, "95 % target", color=INK2)
a.axhline(75, color=CRIT, ls=":", lw=1); a.text(3.9, 70, "plateau 75 %", color=CRIT)
a.plot([], [], color=BLUE, label="reached Run (11)"); a.plot([], [], color=CRIT, label="timed out → FsmError (7)")
a.legend(loc="center right"); a.set_xlabel("s since precharge start"); a.set_ylabel("DC bus / pack (%)")
a.set_title("A · Precharge: bimodal, failures settle at exactly 75 %")

# B open-wire event 0098
a = ax[0, 1]; d = L(98); s = d[(d.tick_ms >= 95000) & (d.tick_ms <= 109000)]; t = s.tick_ms / 1000
a.plot(t, s.pack_mV / 1000, color=BLUE, label="pack voltage (V)")
a.plot(t, s.I_raw_mA / 1000 + 100, color=ORANGE, lw=1.2, label="I_raw + 100 (A)")
a.axvline(106.25, color=CRIT, lw=1); a.text(106.35, 200, "CellOpenWire\nmask 31\n(all modules)", color=CRIT)
a.annotate("one poll: all 95 cells\n+7–10 %, I ≈ 0", xy=(106.5, 374.5), xytext=(97, 368), color=INK,
           arrowprops=dict(arrowstyle="->", color=INK2))
a.legend(loc="lower left"); a.set_xlabel("tick (s), LOG0098"); a.set_ylabel("V  /  A offset by +100")
a.set_title("B · OpenWire trip = pack-wide measurement glitch")

# C C1 artifact
a = ax[1, 0]; d = L(80); s = d[(d.tick_ms >= 20000) & (d.tick_ms <= 26000)]; t = s.tick_ms / 1000
for c, col, lab in (("c0_0", BLUE, "c0_0 (LTC1 C1)"), ("c0_9", ORANGE, "c0_9 (LTC2 C1)"), ("c0_1", AQUA, "c0_1 (LTC1 C2)")):
    a.step(t, s[c], where="post", color=col, label=lab)
a.legend(loc="upper right", ncol=3); a.set_xlabel("tick (s), LOG0080, idle, I ≈ 0"); a.set_ylabel("cell voltage (mV)")
a.set_ylim(3765, 3845); a.set_title("C · C1 of every LTC toggles ~38 mV, 1 s period (all 10 ICs)")

# D over-current 0091
a = ax[1, 1]; d = L(91); s = d[(d.tick_ms >= 50000) & (d.tick_ms <= 65000)]; t = s.tick_ms / 1000
a.plot(t, s.I_raw_mA / 1000, color=ORANGE, lw=1.1, label="I_raw")
a.plot(t, s.I_filt_mA / 1000, color=BLUE, label="I_filt (trip input)")
a.axhline(185, color=CRIT, ls="--", lw=1); a.text(50.3, 190, "185 A limit", color=CRIT)
a.legend(loc="upper left"); a.set_xlabel("tick (s), LOG0091"); a.set_ylabel("A")
a.set_title("D · Genuine over-current: 245 A raw, vmin 2873 mV")

# E IMU at TS-off
a = ax[2, 0]
for i, t0, col, lab in ((93, 148.12, CRIT, "0093 → ModOffline"), (74, 13.16, ORANGE, "0074 → NtcDisc"), (106, 81.79, BLUE, "0106 → no fault")):
    m = I_(i); tt = m.tick_ms / 1000; w = (tt >= t0 - 0.5) & (tt <= t0 + 0.6)
    acc = m[["ax_g", "ay_g", "az_g"]].values
    da = np.r_[0, np.linalg.norm(np.diff(acc, axis=0), axis=1)]
    a.plot(tt[w] - t0, da[w], color=col, label=lab, lw=1.2)
a.axvspan(0.13, 0.38, color=GRID, alpha=0.8); a.text(0.14, 0.072, "latched in\nnext log row", color=INK2)
a.legend(loc="upper left"); a.set_xlabel("s relative to AIR-open clunk"); a.set_ylabel("|Δa| per 10 ms (g)")
a.set_title("E · IMU: AIR opening, then BMS glitch within ≤ 0.4 s")

# F rest cell map
a = ax[2, 1]; d = L(106); r = d[(d.fsm == 0) & (d.tick_ms < 11000)].iloc[4:]
V = np.array([[r[f"c{m}_{n}"].mean() for n in range(19)] for m in range(5)])
im = a.imshow(V - np.median(V), cmap="RdBu", vmin=-200, vmax=200, aspect="auto")
a.set_yticks(range(5), [f"M{m}" for m in range(5)]); a.set_xticks(range(19)); a.grid(False)
a.axvline(8.5, color=INK, lw=0.8)
for m in range(5):
    for n in range(19):
        a.text(n, m, f"{V[m, n] - np.median(V):+.0f}", ha="center", va="center", fontsize=6.5, color=INK)
fig.colorbar(im, ax=a, fraction=0.04, pad=0.01, label="mV vs pack median")
a.set_xlabel("cell index (0–8 LTC1 | 9–18 LTC2)")
a.set_title(f"F · Rest imbalance, LOG0106 (median {np.median(V):.0f} mV, spread {V.max()-V.min():.0f} mV)")
fig.savefig(f"{OUT}/ams_diagnosis.png", dpi=130)
print("ok")

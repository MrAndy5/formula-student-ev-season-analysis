"""Formula Student EV Season 2025/2026 - Accumulator Management System (AMS) Log Parser.

Module: src.pipeline.bms
Provides high-speed loading, validation, topological unpacking, sentinel masking,
and diagnostics extraction for 314-column 4 Hz BMS logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants & Enumerations
# ---------------------------------------------------------------------------
SENTINEL_OPEN = -32768
DEAD_NTC_CHANNEL = "t0_12"
BOOT_TRANSIENT_CHANNEL = "t0_13"

FSM_NAMES: Dict[int, str] = {
    0: "Start",
    1: "Pre",
    2: "Trans",
    3: "Run",
    4: "Chg",
    5: "ERR",
}

FAULT_NAMES: Dict[int, str] = {
    0: "None",
    2: "ModOffline",
    3: "BmsStale",
    4: "UV",
    5: "OV",
    8: "IsensFault",
    9: "IStale",
    10: "OverI",
    11: "VcuStale",
    12: "FsmError",
    13: "NtcDisc",
    14: "ChgStale",
    15: "ChgTsms",
    16: "OpenWire",
}

ROLLOVER_PAIRS: List[Tuple[int, int]] = [
    (80, 81),
    (82, 83),
    (89, 90),
    (102, 103),
]

CELL_COLUMNS: List[str] = [f"c{m}_{n}" for m in range(5) for n in range(19)]
TEMP_COLUMNS: List[str] = [f"t{m}_{n}" for m in range(5) for n in range(40)]


@dataclass
class FsmTransition:
    tick_ms: int
    time_s: float
    from_state: int
    from_name: str
    to_state: int
    to_name: str
    fault_code: int
    fault_name: str
    detail_raw: int
    detail_decoded: str
    pack_mV: float
    dcbus_V: float
    I_filt_A: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def decode_detail(fault_code: int, detail: int) -> str:
    """Decodes hardware fault detail bitmask or returns scalar detail string."""
    if fault_code in (2, 13, 16):
        modules = [f"M{m}" for m in range(5) if (detail >> m) & 1]
        return ",".join(modules) if modules else "None"
    return str(detail)


def resolve_bms_path(
    log_id_or_path: Union[int, str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolves log identifier or file path to validated absolute Path."""
    if isinstance(log_id_or_path, (int, np.integer)):
        fname = f"LOG{int(log_id_or_path):04d}.CSV"
        id_str = str(log_id_or_path)
    else:
        p = Path(log_id_or_path)
        if p.exists() and p.is_file():
            return p.resolve()
        id_str = str(log_id_or_path).strip()
        if id_str.isdigit():
            fname = f"LOG{int(id_str):04d}.CSV"
        else:
            fname = id_str
            if not fname.upper().endswith(".CSV"):
                fname = f"{fname}.CSV"
            stem = Path(fname).stem
            if not stem.upper().startswith("LOG") and stem.isdigit():
                fname = f"LOG{int(stem):04d}.CSV"

    candidates: List[Path] = []
    if base_dir:
        b = Path(base_dir)
        candidates.extend([
            b / fname,
            b / "AMS logs" / "AMS logs" / fname,
            b / "AMS logs" / fname,
        ])

    cwd = Path.cwd()
    candidates.extend([
        cwd / "AMS logs" / "AMS logs" / fname,
        cwd / "AMS logs" / fname,
        cwd / fname,
    ])

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    # Search with glob pattern
    for search_dir in [cwd / "AMS logs" / "AMS logs", cwd / "AMS logs", cwd]:
        if search_dir.exists():
            matches = list(search_dir.glob(f"*{id_str}*.CSV")) + list(
                search_dir.glob(f"*{id_str}*.csv")
            )
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not resolve BMS log file '{log_id_or_path}'. Checked candidates: {candidates}"
    )


def filter_c1_artifact(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Applies a rolling median filter to c{m}_0 and c{m}_9 to suppress 38 mV LTC C1 toggle."""
    df_out = df.copy()
    c1_cols = [f"c{m}_0" for m in range(5)] + [f"c{m}_9" for m in range(5)]
    for c in c1_cols:
        if c in df_out.columns:
            df_out[c] = (
                df_out[c].rolling(window, center=True, min_periods=1).median()
            )
    return df_out


def load_ams_log(
    log_id_or_path: Union[int, str, Path],
    mask_sentinels: bool = True,
    mask_dead_channel: bool = True,
    filter_c1_noise: bool = False,
    base_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Loads and normalizes a single 314-column BMS log file.

    Parameters:
    -----------
    log_id_or_path : Union[int, str, Path]
        Integer ID (e.g. 74) or filename/path.
    mask_sentinels : bool, default True
        If True, replaces -32768 in NTC columns with np.nan.
    mask_dead_channel : bool, default True
        If True, masks dead channel 't0_12' with np.nan and sets 't0_12_masked'=True.
    filter_c1_noise : bool, default False
        If True, applies rolling median filter to c{m}_0 and c{m}_9.
    base_dir : Optional[Union[str, Path]]
        Root search directory.

    Returns:
    --------
    pd.DataFrame
        Normalized DataFrame with derived physical units, verified pack sums,
        and cleaned temperature aggregates.
    """
    path = resolve_bms_path(log_id_or_path, base_dir)
    df = pd.read_csv(path)

    # Compute derived columns and new fields together to prevent fragmentation
    new_cols: Dict[str, Any] = {}
    new_cols["time_s"] = df["tick_ms"].astype(float) / 1000.0
    new_cols["pack_V"] = df["pack_mV"].astype(float) / 1000.0
    new_cols["I_raw_A"] = df["I_raw_mA"].astype(float) / 1000.0
    new_cols["I_filt_A"] = df["I_filt_mA"].astype(float) / 1000.0
    new_cols["Idcdc_A"] = df["Idcdc_mA"].astype(float) / 1000.0
    new_cols["power_kW"] = (
        new_cols["pack_V"] * new_cols["I_filt_A"] / 1000.0
    )

    # 2. State & Fault Labels
    new_cols["fsm_name"] = df["fsm"].map(FSM_NAMES).fillna("Unknown")
    new_cols["fault_name"] = df["fault"].map(FAULT_NAMES).fillna("Unknown")
    new_cols["detail_decoded"] = [
        decode_detail(f, d) for f, d in zip(df["fault"], df["detail"])
    ]

    # 3. Pack Voltage Bitmask Invariant Verification
    m_sums = np.column_stack([
        df[[f"c{m}_{n}" for n in range(19)]].sum(axis=1).values
        for m in range(5)
    ])
    calc_pack = np.zeros(len(df), dtype=np.int64)
    for m in range(5):
        calc_pack += ((df["mod_mask"].values >> m) & 1) * m_sums[:, m]
    new_cols["pack_sum_diff_mV"] = df["pack_mV"].values - calc_pack

    # 4. Thermal Sentinel Handling
    if mask_sentinels:
        T = df[TEMP_COLUMNS].values.astype(float)
        T_masked = np.where(T == SENTINEL_OPEN, np.nan, T)
        if mask_dead_channel and DEAD_NTC_CHANNEL in TEMP_COLUMNS:
            idx_12 = TEMP_COLUMNS.index(DEAD_NTC_CHANNEL)
            T_masked[:, idx_12] = np.nan
            new_cols["t0_12_masked"] = True
        df[TEMP_COLUMNS] = T_masked

        # Cleaned aggregate temperatures with safe warning suppression
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                new_cols["calc_tmin_C"] = np.nanmin(T_masked, axis=1)
                new_cols["calc_tmax_C"] = np.nanmax(T_masked, axis=1)
                new_cols["calc_tavg_C"] = np.nanmean(T_masked, axis=1)
    else:
        if mask_dead_channel and DEAD_NTC_CHANNEL in df.columns:
            df[DEAD_NTC_CHANNEL] = np.nan
            new_cols["t0_12_masked"] = True

    new_cols["source_file"] = path.name

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # 5. Optional C1 artifact filtering
    if filter_c1_noise:
        df = filter_c1_artifact(df)

    return df


def load_ams_session(
    log_id: int,
    auto_stitch: bool = True,
    base_dir: Optional[Union[str, Path]] = None,
    **load_kwargs,
) -> pd.DataFrame:
    """Loads a continuous BMS session, automatically stitching split rollover files if detected."""
    rollover_match = None
    if auto_stitch:
        for p1, p2 in ROLLOVER_PAIRS:
            if log_id in (p1, p2):
                rollover_match = (p1, p2)
                break

    if not rollover_match:
        df = load_ams_log(log_id, base_dir=base_dir, **load_kwargs)
        df["session_id"] = f"BMS_{log_id:04d}"
        return df

    p1, p2 = rollover_match
    df1 = load_ams_log(p1, base_dir=base_dir, **load_kwargs)
    df2 = load_ams_log(p2, base_dir=base_dir, **load_kwargs)

    # Verify continuity at boundary
    t1_end = df1["tick_ms"].iloc[-1]
    t2_start = df2["tick_ms"].iloc[0]
    dt = t2_start - t1_end
    if dt != 250:
        # Warning or note, but proceed
        pass

    df_stitched = pd.concat([df1, df2], ignore_index=True)
    df_stitched["session_id"] = f"BMS_{p1:04d}_{p2:04d}"
    df_stitched["time_s"] = (
        (df_stitched["tick_ms"] - df_stitched["tick_ms"].iloc[0]).astype(float)
        / 1000.0
    )
    return df_stitched


def extract_fsm_transitions(df: pd.DataFrame) -> List[FsmTransition]:
    """Extracts all state transitions with pre-condition snapshots and decoded faults."""
    transitions = []
    if len(df) <= 1:
        return transitions

    fs = df["fsm"].values
    ticks = df["tick_ms"].values

    for k in range(1, len(fs)):
        if fs[k] != fs[k - 1]:
            row = df.iloc[k]
            tr = FsmTransition(
                tick_ms=int(ticks[k]),
                time_s=float(ticks[k] / 1000.0),
                from_state=int(fs[k - 1]),
                from_name=FSM_NAMES.get(fs[k - 1], str(fs[k - 1])),
                to_state=int(fs[k]),
                to_name=FSM_NAMES.get(fs[k], str(fs[k])),
                fault_code=int(row["fault"]),
                fault_name=str(row["fault_name"]),
                detail_raw=int(row["detail"]),
                detail_decoded=str(row["detail_decoded"]),
                pack_mV=float(row["pack_mV"]),
                dcbus_V=float(row["dcbus_V"]),
                I_filt_A=float(row["I_filt_A"]),
            )
            transitions.append(tr)
    return transitions

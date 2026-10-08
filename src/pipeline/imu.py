"""Formula Student EV Season 2025/2026 - 100 Hz Inertial Measurement Unit (IMU) Processor.

Module: src.pipeline.imu
Provides high-rate 6-DOF data loading, Butterworth low-pass filtering, mechanical shock
detection (|Delta a| > 0.05 g), and time synchronization with co-located BMS logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Constants & Thresholds
# ---------------------------------------------------------------------------
IMU_FS_HZ = 100.0
IMU_DT_S = 0.010
SHOCK_THRESHOLD_G = 0.05  # Jerk threshold |Δa| > 0.05 g per 10 ms
SHOCK_CLUSTER_WINDOW_MS = 200  # Cluster window for discrete shock events
BUTTER_CUTOFF_HZ = 10.0  # Cutoff frequency for body acceleration LP filter

IMU_SESSIONS = [74, 93, 94, 97, 100, 101, 105, 106]


@dataclass
class ShockEvent:
    shock_id: int
    t_start_ms: int
    t_end_ms: int
    duration_ms: int
    peak_da_g: float
    peak_a_mag_g: float
    sample_count: int
    dominant_axis: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_imu_path(
    log_id_or_path: Union[int, str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolves IMU identifier or filename to validated absolute Path."""
    if isinstance(log_id_or_path, (int, np.integer)):
        fname = f"IMU{int(log_id_or_path):04d}.CSV"
        id_str = str(log_id_or_path)
    else:
        p = Path(log_id_or_path)
        if p.exists() and p.is_file():
            return p.resolve()
        id_str = str(log_id_or_path).strip()
        if id_str.isdigit():
            fname = f"IMU{int(id_str):04d}.CSV"
        else:
            fname = id_str
            if not fname.upper().endswith(".CSV"):
                fname = f"{fname}.CSV"
            stem = Path(fname).stem
            if not stem.upper().startswith("IMU") and stem.isdigit():
                fname = f"IMU{int(stem):04d}.CSV"

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
        f"Could not resolve IMU log file '{log_id_or_path}'. Checked candidates: {candidates}"
    )


def load_imu_log(
    log_id_or_path: Union[int, str, Path],
    apply_butterworth: bool = True,
    cutoff_hz: float = BUTTER_CUTOFF_HZ,
    base_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Loads and calibrates 100 Hz IMU telemetry.

    Parameters:
    -----------
    log_id_or_path : Union[int, str, Path]
        Integer ID (e.g. 74) or filename/path.
    apply_butterworth : bool, default True
        If True, applies 2nd-order zero-phase 10 Hz LP filter to acc and gyro.
    cutoff_hz : float, default 10.0
        Butterworth cutoff frequency.
    base_dir : Optional[Union[str, Path]]
        Root search directory.

    Returns:
    --------
    pd.DataFrame
        Calibrated DataFrame containing raw 6-DOF channels, filtered channels,
        jerk magnitude (|Δa|), and dynamic body acceleration.
    """
    path = resolve_imu_path(log_id_or_path, base_dir)
    df = pd.read_csv(path)

    new_cols: Dict[str, Any] = {}
    new_cols["time_s"] = df["tick_ms"].astype(float) / 1000.0

    # 1. Total vector norms
    acc = df[["ax_g", "ay_g", "az_g"]].values.astype(float)
    gyro = df[["gx_rad_s", "gy_rad_s", "gz_rad_s"]].values.astype(float)
    new_cols["a_mag_g"] = np.linalg.norm(acc, axis=1)
    new_cols["g_mag_rad_s"] = np.linalg.norm(gyro, axis=1)

    # 2. Sample-to-sample jerk magnitude (|Δa|)
    da = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        da[1:] = np.linalg.norm(np.diff(acc, axis=0), axis=1)
    new_cols["da_mag_g"] = da

    # Per-axis delta
    da_xyz = np.zeros_like(acc)
    if len(df) > 1:
        da_xyz[1:] = np.diff(acc, axis=0)
    new_cols["da_x_g"] = da_xyz[:, 0]
    new_cols["da_y_g"] = da_xyz[:, 1]
    new_cols["da_z_g"] = da_xyz[:, 2]

    # 3. 6-DOF Butterworth Low-Pass Filtering
    if apply_butterworth and len(df) > 15:
        nyq = 0.5 * IMU_FS_HZ
        b, a = butter(2, cutoff_hz / nyq, btype="low")
        acc_filt = filtfilt(b, a, acc, axis=0)
        gyro_filt = filtfilt(b, a, gyro, axis=0)

        new_cols["ax_filt_g"] = acc_filt[:, 0]
        new_cols["ay_filt_g"] = acc_filt[:, 1]
        new_cols["az_filt_g"] = acc_filt[:, 2]
        new_cols["gx_filt_rad_s"] = gyro_filt[:, 0]
        new_cols["gy_filt_rad_s"] = gyro_filt[:, 1]
        new_cols["gz_filt_rad_s"] = gyro_filt[:, 2]
        new_cols["a_dyn_mag_g"] = np.linalg.norm(acc_filt, axis=1)

        # Static gravity compensation (first 200 samples ~ 2.0 s)
        n_stat = min(200, len(df))
        g_static = acc[:n_stat].mean(axis=0)
        new_cols["ax_dyn_g"] = acc_filt[:, 0] - g_static[0]
        new_cols["ay_dyn_g"] = acc_filt[:, 1] - g_static[1]
        new_cols["az_dyn_g"] = acc_filt[:, 2] - g_static[2]
    else:
        new_cols["ax_filt_g"] = acc[:, 0]
        new_cols["ay_filt_g"] = acc[:, 1]
        new_cols["az_filt_g"] = acc[:, 2]
        new_cols["gx_filt_rad_s"] = gyro[:, 0]
        new_cols["gy_filt_rad_s"] = gyro[:, 1]
        new_cols["gz_filt_rad_s"] = gyro[:, 2]
        new_cols["a_dyn_mag_g"] = new_cols["a_mag_g"]
        new_cols["ax_dyn_g"] = acc[:, 0]
        new_cols["ay_dyn_g"] = acc[:, 1]
        new_cols["az_dyn_g"] = acc[:, 2] - 1.0

    new_cols["source_file"] = path.name

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def detect_mechanical_shocks(
    df: pd.DataFrame,
    threshold_g: float = SHOCK_THRESHOLD_G,
    cluster_window_ms: int = SHOCK_CLUSTER_WINDOW_MS,
) -> List[ShockEvent]:
    """Detects and clusters high-jerk mechanical shock events (|Δa| > 0.05 g)."""
    if len(df) == 0:
        return []

    tm = df["tick_ms"].values
    da = df["da_mag_g"].values if "da_mag_g" in df.columns else np.zeros(len(df))
    a_mag = df["a_mag_g"].values if "a_mag_g" in df.columns else np.zeros(len(df))
    da_xyz = (
        np.abs(df[["da_x_g", "da_y_g", "da_z_g"]].values)
        if "da_x_g" in df.columns
        else np.zeros((len(df), 3))
    )

    shock_idx = np.where(da > threshold_g)[0]
    if len(shock_idx) == 0:
        return []

    clusters: List[Dict[str, Any]] = []
    for idx in shock_idx:
        t_event = int(tm[idx])
        if (
            not clusters
            or (t_event - clusters[-1]["t_end_ms"]) > cluster_window_ms
        ):
            clusters.append({
                "t_start_ms": t_event,
                "t_end_ms": t_event,
                "peak_da_g": float(da[idx]),
                "peak_a_mag_g": float(a_mag[idx]),
                "sample_count": 1,
                "axis_contrib": da_xyz[idx].copy(),
            })
        else:
            clusters[-1]["t_end_ms"] = t_event
            clusters[-1]["peak_da_g"] = max(
                clusters[-1]["peak_da_g"], float(da[idx])
            )
            clusters[-1]["peak_a_mag_g"] = max(
                clusters[-1]["peak_a_mag_g"], float(a_mag[idx])
            )
            clusters[-1]["sample_count"] += 1
            clusters[-1]["axis_contrib"] += da_xyz[idx]

    events: List[ShockEvent] = []
    axes = ["X", "Y", "Z"]
    for i, c in enumerate(clusters):
        dom_axis = axes[int(np.argmax(c["axis_contrib"]))]
        ev = ShockEvent(
            shock_id=i + 1,
            t_start_ms=int(c["t_start_ms"]),
            t_end_ms=int(c["t_end_ms"]),
            duration_ms=int(c["t_end_ms"] - c["t_start_ms"]),
            peak_da_g=float(c["peak_da_g"]),
            peak_a_mag_g=float(c["peak_a_mag_g"]),
            sample_count=int(c["sample_count"]),
            dominant_axis=dom_axis,
        )
        events.append(ev)
    return events


def correlate_imu_bms(
    imu_df: pd.DataFrame,
    bms_df: pd.DataFrame,
    window_ms: int = 500,
) -> pd.DataFrame:
    """Correlates BMS state machine transitions with high-rate IMU shocks,

    quantifying time lag between contactor opening and false glitch latching.
    """
    from src.pipeline.bms import extract_fsm_transitions

    transitions = extract_fsm_transitions(bms_df)
    shocks = detect_mechanical_shocks(imu_df)

    records = []
    for tr in transitions:
        t_tr = tr.tick_ms
        nearby_shocks = [
            s for s in shocks if abs(s.t_start_ms - t_tr) <= window_ms
        ]

        nearest_shock = None
        min_dt = 1e9
        for s in nearby_shocks:
            dt = t_tr - s.t_start_ms
            if abs(dt) < abs(min_dt):
                min_dt = dt
                nearest_shock = s

        records.append({
            "bms_tick_ms": t_tr,
            "bms_time_s": tr.time_s,
            "fsm_transition": f"{tr.from_name}->{tr.to_name}",
            "fault_name": tr.fault_name,
            "detail": tr.detail_decoded,
            "has_nearby_shock": nearest_shock is not None,
            "shock_peak_da_g": nearest_shock.peak_da_g if nearest_shock else 0.0,
            "shock_tick_ms": nearest_shock.t_start_ms if nearest_shock else None,
            "time_lag_ms": min_dt if nearest_shock else None,
        })
    return pd.DataFrame(records)


def resample_bms_to_imu(
    bms_df: pd.DataFrame, imu_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges 4 Hz BMS scalars onto the 100 Hz IMU timeline via backward merge_asof."""
    bms_cols = [
        "tick_ms",
        "fsm",
        "fsm_name",
        "ams_ok",
        "fault",
        "fault_name",
        "pack_V",
        "I_filt_A",
        "power_kW",
        "dcbus_V",
        "vmin_mV",
        "vmax_mV",
    ]
    available_cols = [c for c in bms_cols if c in bms_df.columns]

    merged = pd.merge_asof(
        imu_df.sort_values("tick_ms"),
        bms_df[available_cols].sort_values("tick_ms"),
        on="tick_ms",
        direction="backward",
    )
    return merged

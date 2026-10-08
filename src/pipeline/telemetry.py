"""Formula Student EV Season 2025/2026 - Vehicle Telemetry Ingestion & Normalizer.

Module: src.pipeline.telemetry
Implements loading, schema adaptation (Schemas 0-9), speed calibration,
DCDC current normalization, continuous time vector extraction, and canonical signal mapping.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.parse
import numpy as np
import pandas as pd

# Calibration Constants
SPEED_CALIBRATION_DIVISOR = 37.978  # v_kmh = inv_rpm / 37.978
APPS1_IDLE_RAW = 2498.0
APPS1_MAX_RAW = 3405.0
APPS2_IDLE_RAW = 2332.0
APPS2_MAX_RAW = 3025.0
BRAKE_RELEASED_RAW = 550.0
BRAKE_LOCK_RAW = 2700.0

CANONICAL_COLUMNS = [
    "timestamp_s",
    "inv_rpm",
    "speed_kmh",
    "torque_actual_pct",
    "v_dcbus_V",
    "i_batt_A",
    "i_dcdc_A",
    "apps1_pct",
    "apps2_pct",
    "brake_pct",
    "ams_fsm_state",
    "ctrl_state",
    "rule_ev_2_3_flag",
    "rule_t11_8_9_flag",
]

EMPTY_STUB_MAX_BYTES = 850


def detect_telemetry_schema(df: pd.DataFrame) -> int:
    """Identify which of the 10 telemetry schemas (0 to 9) a DataFrame belongs to."""
    cols = set(df.columns)
    col_count = len(df.columns)

    if "susp_force_fl" in cols or "dc_bus_voltage" in cols:
        return 6  # Legacy 44-col simulator (Dec 2025)
    if "ams_t_mod0_cell0" in cols:
        return 8  # 153-col cell temperature simulation (July 9)
    if "gps_sog_knots" in cols:
        return 7 if col_count == 50 else 9  # 50-col vs 59-col demo
    if "_source" in cols:
        return 3 if col_count == 63 else 5  # 63-col vs 70-col merged
    if "eff_wh_min" in cols or col_count == 69:
        return 4  # 69-col Barcelona/Autumn standard
    if "gps_lat_deg" in cols or col_count == 62:
        return 2  # 62-col Austria standard
    if "dem_code" in cols or col_count == 56:
        return 1  # 56-col Cantoblanco upgrade
    if col_count == 53:
        return 0  # 53-col Cantoblanco baseline

    count_map = {
        53: 0,
        56: 1,
        62: 2,
        63: 3,
        69: 4,
        70: 5,
        44: 6,
        50: 7,
        153: 8,
        59: 9,
    }
    return count_map.get(col_count, -1)


def extract_standardized_time(df: pd.DataFrame) -> pd.Series:
    """Extract continuous, monotonic timestamp_s starting at 0.0s."""
    if len(df) == 0:
        return pd.Series(dtype=float)

    if "time" in df.columns:
        try:
            t_dt = pd.to_datetime(df["time"])
            if not t_dt.isna().all():
                diffs = (t_dt - t_dt.iloc[0]).dt.total_seconds().values
                diffs = np.maximum.accumulate(diffs)
                return pd.Series(diffs, index=df.index, dtype=float)
        except Exception:
            pass

    if "time_elapsed_s" in df.columns:
        s = df["time_elapsed_s"].astype(float).values
        diffs = s - s[0]
        diffs = np.maximum.accumulate(diffs)
        return pd.Series(diffs, index=df.index, dtype=float)

    if "tick_ms" in df.columns:
        ticks = df["tick_ms"].astype(float).values
        diffs = (ticks - ticks[0]) / 1000.0
        diffs = np.maximum.accumulate(diffs)
        return pd.Series(diffs, index=df.index, dtype=float)

    return pd.Series(np.arange(len(df)) * 0.2, index=df.index, dtype=float)


def compute_calibrated_speed(df: pd.DataFrame) -> pd.Series:
    """Derive calibrated vehicle speed in km/h: v = inv_rpm / 37.978."""
    if len(df) == 0:
        return pd.Series(dtype=float)

    if "inv_rpm" in df.columns:
        rpm = df["inv_rpm"].astype(float)
    elif "rpm" in df.columns:
        rpm = df["rpm"].astype(float)
    else:
        rpm = pd.Series(0.0, index=df.index, dtype=float)

    speed_kmh = rpm / SPEED_CALIBRATION_DIVISOR
    return speed_kmh.clip(lower=0.0)


def normalize_dcdc_current(series: pd.Series) -> pd.Series:
    """Normalizes DCDC converter current to positive amperes (+37A nominal).

    Resolves July 19 (-385A -> +38.5A) and July 20 (-38A -> +38A).
    """
    s = series.astype(float).copy()
    nonzero = s[s != 0.0]
    if len(nonzero) == 0:
        return s.clip(lower=0.0)

    med = nonzero.median()
    if med < -100.0:
        # July 19: 10x scaled and negative polarity
        return (-s / 10.0).clip(lower=0.0)
    elif med < -5.0:
        # July 20: negative polarity
        return (-s).clip(lower=0.0)
    else:
        # July 21 onward: nominal positive
        return s.clip(lower=0.0)


def normalize_pedals(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Normalizes apps1_pct, apps2_pct, and brake_pct into 0.0 to 100.0%."""
    if len(df) == 0:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    # APPS 1
    if "apps1_raw" in df.columns:
        apps1 = (
            (df["apps1_raw"].astype(float) - APPS1_IDLE_RAW)
            / (APPS1_MAX_RAW - APPS1_IDLE_RAW)
            * 100.0
        ).clip(0.0, 100.0)
    elif "throttle_pct" in df.columns:
        apps1 = df["throttle_pct"].astype(float).clip(0.0, 100.0)
    else:
        apps1 = pd.Series(0.0, index=df.index, dtype=float)

    # APPS 2
    if "apps2_raw" in df.columns:
        apps2 = (
            (df["apps2_raw"].astype(float) - APPS2_IDLE_RAW)
            / (APPS2_MAX_RAW - APPS2_IDLE_RAW)
            * 100.0
        ).clip(0.0, 100.0)
    elif "throttle_pct" in df.columns:
        apps2 = df["throttle_pct"].astype(float).clip(0.0, 100.0)
    else:
        apps2 = pd.Series(0.0, index=df.index, dtype=float)

    # Brake
    if "brake_raw" in df.columns:
        brake = (
            (df["brake_raw"].astype(float) - BRAKE_RELEASED_RAW)
            / (BRAKE_LOCK_RAW - BRAKE_RELEASED_RAW)
            * 100.0
        ).clip(0.0, 100.0)
    elif "brake_pct" in df.columns:
        brake = df["brake_pct"].astype(float).clip(0.0, 100.0)
    else:
        brake = pd.Series(0.0, index=df.index, dtype=float)

    return apps1, apps2, brake


def normalize_telemetry_dataframe(
    df_raw: pd.DataFrame,
    source_filename: str = "",
) -> pd.DataFrame:
    """Transforms raw telemetry DataFrame into canonical clean telemetry DataFrame."""
    dtypes = {
        "timestamp_s": float,
        "inv_rpm": float,
        "speed_kmh": float,
        "torque_actual_pct": float,
        "v_dcbus_V": float,
        "i_batt_A": float,
        "i_dcdc_A": float,
        "apps1_pct": float,
        "apps2_pct": float,
        "brake_pct": float,
        "ams_fsm_state": int,
        "ctrl_state": int,
        "rule_ev_2_3_flag": int,
        "rule_t11_8_9_flag": int,
    }

    if len(df_raw) == 0:
        empty_df = pd.DataFrame(
            {col: pd.Series(dtype=dt) for col, dt in dtypes.items()}
        )
        if source_filename:
            empty_df["source_file"] = pd.Series(dtype=str)
        return empty_df

    out = pd.DataFrame(index=df_raw.index)

    # 1. timestamp_s
    out["timestamp_s"] = extract_standardized_time(df_raw)

    # 2. inv_rpm & speed_kmh
    if "inv_rpm" in df_raw.columns:
        out["inv_rpm"] = df_raw["inv_rpm"].astype(float)
    elif "rpm" in df_raw.columns:
        out["inv_rpm"] = df_raw["rpm"].astype(float)
    else:
        out["inv_rpm"] = 0.0
    out["speed_kmh"] = compute_calibrated_speed(df_raw)

    # 3. torque_actual_pct
    if "torque_pct" in df_raw.columns:
        out["torque_actual_pct"] = (
            df_raw["torque_pct"].astype(float).clip(0.0, 100.0)
        )
    elif "torque_req" in df_raw.columns:
        out["torque_actual_pct"] = (
            df_raw["torque_req"].astype(float).clip(0.0, 100.0)
        )
    elif "throttle_pct" in df_raw.columns:
        out["torque_actual_pct"] = (
            df_raw["throttle_pct"].astype(float).clip(0.0, 100.0)
        )
    else:
        out["torque_actual_pct"] = 0.0

    # 4. v_dcbus_V
    if "inv_dc_bus_V" in df_raw.columns:
        out["v_dcbus_V"] = df_raw["inv_dc_bus_V"].astype(float)
    elif "dc_bus_voltage" in df_raw.columns:
        out["v_dcbus_V"] = df_raw["dc_bus_voltage"].astype(float)
    elif "ams_stack_v" in df_raw.columns:
        out["v_dcbus_V"] = df_raw["ams_stack_v"].astype(float)
    else:
        out["v_dcbus_V"] = 0.0

    # 5. i_batt_A
    if "corriente_accu" in df_raw.columns:
        out["i_batt_A"] = df_raw["corriente_accu"].astype(float)
    elif "ams_current_a" in df_raw.columns:
        out["i_batt_A"] = df_raw["ams_current_a"].astype(float)
    else:
        out["i_batt_A"] = 0.0

    # 6. i_dcdc_A
    if "corriente_dcdc" in df_raw.columns:
        out["i_dcdc_A"] = normalize_dcdc_current(df_raw["corriente_dcdc"])
    else:
        out["i_dcdc_A"] = 0.0

    # 7-10. apps1_pct, apps2_pct, brake_pct
    apps1, apps2, brake = normalize_pedals(df_raw)
    out["apps1_pct"] = apps1
    out["apps2_pct"] = apps2
    out["brake_pct"] = brake

    # 11-12. ams_fsm_state & ctrl_state
    if "ams_fsm_state" in df_raw.columns:
        out["ams_fsm_state"] = df_raw["ams_fsm_state"].fillna(0).astype(int)
    else:
        out["ams_fsm_state"] = 3  # Nominal run default for simulators

    if "ctrl_state" in df_raw.columns:
        out["ctrl_state"] = df_raw["ctrl_state"].fillna(0).astype(int)
    else:
        out["ctrl_state"] = 1  # Drive state default for simulators

    # 13-14. rule flags
    if "ev_2_3" in df_raw.columns:
        out["rule_ev_2_3_flag"] = df_raw["ev_2_3"].fillna(0).astype(int)
    else:
        out["rule_ev_2_3_flag"] = 0

    if "t11_8_9" in df_raw.columns:
        out["rule_t11_8_9_flag"] = df_raw["t11_8_9"].fillna(0).astype(int)
    else:
        out["rule_t11_8_9_flag"] = 0

    # Auxiliary channels preserved for diagnostics and downstream modules
    if "inv_state" in df_raw.columns:
        out["inv_state"] = df_raw["inv_state"].fillna(0).astype(int)
    if "inv_error" in df_raw.columns:
        out["inv_error"] = df_raw["inv_error"].fillna(0).astype(int)
    if "dem_code" in df_raw.columns:
        out["dem_code"] = df_raw["dem_code"].fillna(0).astype(int)
    if "ok_precharge" in df_raw.columns:
        out["ok_precharge"] = df_raw["ok_precharge"].fillna(0).astype(int)
    if "inv_current_actual" in df_raw.columns:
        out["inv_current_phase_A"] = df_raw["inv_current_actual"].astype(float)
    elif "i_actual" in df_raw.columns:
        out["inv_current_phase_A"] = df_raw["i_actual"].astype(float)

    if "soc" in df_raw.columns:
        out["soc"] = df_raw["soc"].fillna(0).astype(int)
    if "v_cell_min_mV" in df_raw.columns:
        out["v_cell_min_mV"] = df_raw["v_cell_min_mV"].fillna(0).astype(int)

    # Inverter & DCDC Temperatures
    if "inv_temp_motor1" in df_raw.columns:
        out["temp_motor_C"] = df_raw["inv_temp_motor1"].astype(float)
    if "inv_temp_pwrstg" in df_raw.columns:
        out["temp_inverter_C"] = df_raw["inv_temp_pwrstg"].astype(float)
    if "temp_dcdc" in df_raw.columns:
        out["temp_dcdc_C"] = df_raw["temp_dcdc"].astype(float)

    # Module min/max voltages & temperatures
    for m in range(5):
        if f"vmin_mod{m}" in df_raw.columns:
            out[f"vmin_mod{m}"] = df_raw[f"vmin_mod{m}"].fillna(0).astype(int)
        if f"vmax_mod{m}" in df_raw.columns:
            out[f"vmax_mod{m}"] = df_raw[f"vmax_mod{m}"].fillna(0).astype(int)
        if f"tmax_mod{m}" in df_raw.columns:
            out[f"tmax_mod{m}"] = df_raw[f"tmax_mod{m}"].fillna(0).astype(int)

    # Raw pedal/brake ADC counts (useful for EV rule checking)
    if "apps1_raw" in df_raw.columns:
        out["apps1_raw"] = df_raw["apps1_raw"].astype(float)
    if "apps2_raw" in df_raw.columns:
        out["apps2_raw"] = df_raw["apps2_raw"].astype(float)
    if "brake_raw" in df_raw.columns:
        out["brake_raw"] = df_raw["brake_raw"].astype(float)

    # GPS and IMU channels if available
    for ch in [
        "gps_lat_deg",
        "gps_lon_deg",
        "gps_speed_kmh",
        "gps_course_deg",
        "gps_sats",
        "gps_has_fix",
        "imu_ax_g",
        "imu_ay_g",
        "imu_az_g",
        "imu_gx_dps",
        "imu_gy_dps",
        "imu_gz_dps",
        "imu_roll_deg",
        "imu_pitch_deg",
        "eff_wh_min",
        "eff_wh_km",
        "batt_r_int",
        "strategy_pwr_target",
        "strategy_rec_torque",
    ]:
        if ch in df_raw.columns:
            out[ch] = df_raw[ch]

    # Provenance and metadata
    out["schema_id"] = detect_telemetry_schema(df_raw)
    if "_source" in df_raw.columns:
        out["_source"] = df_raw["_source"]
    if source_filename:
        out["source_file"] = Path(source_filename).name

    return out


def resolve_telemetry_path(
    identifier: Union[str, int, Path],
    base_dir: str = "datos2526",
) -> str:
    """Resolves session_id, filename, or stem to an absolute CSV filepath.

    Handles URL-encoding (%C3%A9 -> é) and path normalizations.
    """
    id_str = str(identifier).strip()

    # Direct existing path
    if os.path.exists(id_str):
        return os.path.abspath(id_str)

    unquoted = urllib.parse.unquote(id_str)
    quoted = urllib.parse.quote(id_str, safe="()")

    candidates = [
        id_str,
        f"{id_str}.csv",
        f"{id_str}.CSV",
        unquoted,
        f"{unquoted}.csv",
        quoted,
        f"{quoted}.csv",
    ]

    for c in candidates:
        target = os.path.join(base_dir, c)
        if os.path.exists(target):
            return os.path.abspath(target)

    # Glob search for partial matches
    pattern = os.path.join(base_dir, f"*{id_str}*.csv")
    matches = glob.glob(pattern)
    if matches:
        return os.path.abspath(matches[0])

    raise FileNotFoundError(
        f"Telemetry session file not found for identifier: '{identifier}' in '{base_dir}'"
    )


def load_telemetry_file(
    filepath_or_name: Union[str, Path],
    normalize: bool = True,
    base_dir: str = "datos2526",
) -> pd.DataFrame:
    """Loads a single vehicle telemetry CSV file. Handles 0-row empty files gracefully."""
    path = resolve_telemetry_path(filepath_or_name, base_dir=base_dir)

    # Check if empty stub
    file_size = os.path.getsize(path)
    if file_size <= EMPTY_STUB_MAX_BYTES:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) <= 1:
            if normalize:
                return normalize_telemetry_dataframe(pd.DataFrame(), path)
            return pd.read_csv(path)

    df_raw = pd.read_csv(path)
    if normalize:
        return normalize_telemetry_dataframe(df_raw, path)
    return df_raw


def load_session(
    session_id: Union[str, int, Path],
    base_dir: str = "datos2526",
) -> pd.DataFrame:
    """Authoritative interface contract entry point (PROJECT.md line 92).

    Returns a clean, normalized DataFrame containing all mandatory canonical channels.
    """
    return load_telemetry_file(session_id, normalize=True, base_dir=base_dir)

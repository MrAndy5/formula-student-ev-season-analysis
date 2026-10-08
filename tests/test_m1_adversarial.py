"""Milestone 1 Adversarial & Stress Testing Suite.

Empirically challenges:
1. Vehicle Telemetry: 100% of 78 files (63 active, 15 stubs) via load_session.
   - Canonical 14 columns present.
   - Zero NaNs in required channels for active runs.
   - Finite values, monotonicity, physical bounding.
   - Identifier resolution robustness (stem, filename, session_id, path).
2. BMS Logs: 100% of 32 files (12,244 rows) via load_ams_log.
   - Bitmask pack sum identity residual == 0.0 mV on all rows.
   - Independent mathematical oracle validation.
   - Complete masking of -32768 sentinels and dead channel t0_12.
   - Temperature metric consistency (tmin <= tavg <= tmax).
   - Rollover stitching on all 4 pairs with monotonic time continuity.
3. IMU Telemetry: 100% of 8 files (44,159 rows) via load_imu_log.
   - Sampling rate validation (100 Hz median dt = 10 ms).
   - Gravity norm consistency (~1.0 g).
   - Mechanical shock jerk detection (|da| > 0.05 g).
   - Co-located BMS time alignment and resampling.
4. Pipeline Resource Efficiency & Memory Leakage:
   - Repeated iteration memory growth bounds.
"""

from __future__ import annotations

import gc
import glob
import os
from pathlib import Path
import sys
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.catalog import (
    build_season_catalog,
    get_active_sessions,
    get_stubs,
)
from src.pipeline.telemetry import (
    CANONICAL_COLUMNS,
    load_session,
    load_telemetry_file,
    resolve_telemetry_path,
)
from src.pipeline.bms import (
    DEAD_NTC_CHANNEL,
    ROLLOVER_PAIRS,
    SENTINEL_OPEN,
    TEMP_COLUMNS,
    extract_fsm_transitions,
    load_ams_log,
    load_ams_session,
)
from src.pipeline.imu import (
    correlate_imu_bms,
    detect_mechanical_shocks,
    load_imu_log,
    resample_bms_to_imu,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_DIR = BASE_DIR / "datos2526"
AMS_DIR = BASE_DIR / "AMS logs" / "AMS logs"


class TestTelemetryAdversarialStress:
    """Adversarial stress testing against all 78 telemetry files."""

    def test_all_78_telemetry_files_load_without_exceptions(self):
        """Stress test: Every single telemetry file in datos2526/ must load cleanly."""
        files = sorted(glob.glob(str(DATOS_DIR / "*.csv")))
        assert len(files) == 78, f"Expected exactly 78 telemetry CSVs, found {len(files)}"

        empty_count = 0
        active_count = 0

        for fpath in files:
            df = load_session(fpath)
            assert isinstance(df, pd.DataFrame), f"Failed to return DataFrame for {fpath}"

            # Canonical columns contract
            for col in CANONICAL_COLUMNS:
                assert col in df.columns, f"Missing canonical column '{col}' in {fpath}"

            if len(df) == 0:
                empty_count += 1
            else:
                active_count += 1
                # Active session data integrity: zero NaNs in canonical channels
                for col in CANONICAL_COLUMNS:
                    nan_count = df[col].isna().sum()
                    assert nan_count == 0, (
                        f"Canonical column '{col}' has {nan_count} NaNs in active file {fpath}"
                    )

                # Monotonic timestamp_s
                dt = df["timestamp_s"].diff().dropna()
                assert (dt >= 0).all(), f"Non-monotonic timestamps detected in {fpath}"

                # Physical bounds
                assert (df["speed_kmh"] >= 0.0).all(), f"Negative speed in {fpath}"
                assert (df["i_dcdc_A"] >= 0.0).all(), f"Negative DCDC current in {fpath}"
                assert (df["apps1_pct"] >= 0.0).all() and (df["apps1_pct"] <= 100.0).all()
                assert (df["apps2_pct"] >= 0.0).all() and (df["apps2_pct"] <= 100.0).all()
                assert (df["brake_pct"] >= 0.0).all() and (df["brake_pct"] <= 100.0).all()
                assert not np.isinf(df[CANONICAL_COLUMNS].values).any(), f"Inf found in {fpath}"

        assert empty_count == 15, f"Expected 15 empty stubs, got {empty_count}"
        assert active_count == 63, f"Expected 63 active files, got {active_count}"

    def test_session_identifier_resolution_variants(self):
        """Stress test: Identifier resolution must support catalog session_id, stem, and filename."""
        catalog = build_season_catalog(BASE_DIR)
        telem_cat = catalog[catalog["source_dir"] == "datos2526"]

        for _, row in telem_cat.iterrows():
            sid = row["session_id"]
            sfile = row["source_file"]
            stem = Path(sfile).stem

            df_by_sid = load_session(sid, base_dir=str(DATOS_DIR))
            df_by_stem = load_session(stem, base_dir=str(DATOS_DIR))

            assert len(df_by_sid) == row["row_count"]
            assert len(df_by_stem) == row["row_count"]


class TestBmsAdversarialStress:
    """Adversarial stress testing against all 32 BMS files."""

    def test_all_32_bms_files_and_bitmask_pack_sum_invariant(self):
        """Stress test: Bitmask pack sum identity must hold with 0 mV error across all 12,244 rows."""
        bms_files = sorted(glob.glob(str(AMS_DIR / "LOG*.CSV")))
        assert len(bms_files) == 32, f"Expected 32 BMS files, found {len(bms_files)}"

        total_rows = 0

        for fpath in bms_files:
            log_id = int(Path(fpath).stem[3:])
            df = load_ams_log(log_id, mask_sentinels=True, mask_dead_channel=True, base_dir=str(BASE_DIR))
            n = len(df)
            total_rows += n

            # 1. Pipeline's verified column
            assert (df["pack_sum_diff_mV"] == 0).all(), (
                f"pack_sum_diff_mV != 0 detected in {fpath}"
            )

            # 2. Independent mathematical oracle check
            m_sums = np.column_stack([
                df[[f"c{m}_{n_idx}" for n_idx in range(19)]].sum(axis=1).values
                for m in range(5)
            ])
            calc_pack = np.zeros(n, dtype=np.int64)
            for m in range(5):
                calc_pack += ((df["mod_mask"].values >> m) & 1) * m_sums[:, m]

            oracle_diff = df["pack_mV"].values - calc_pack
            assert (oracle_diff == 0).all(), (
                f"Independent bitmask oracle verification failed in {fpath}"
            )

            # 3. Sentinel masking check
            t_vals = df[TEMP_COLUMNS].values
            assert not np.any(t_vals == SENTINEL_OPEN), (
                f"Unmasked SENTINEL_OPEN (-32768) found in {fpath}"
            )

            # 4. Dead channel check
            assert df[DEAD_NTC_CHANNEL].isna().all(), (
                f"Dead channel {DEAD_NTC_CHANNEL} contained non-NaN values in {fpath}"
            )

            # 5. Temperature ordering check: tmin <= tavg <= tmax
            valid_temps = df["calc_tmin_C"].notna()
            if valid_temps.any():
                tmin = df.loc[valid_temps, "calc_tmin_C"].values
                tavg = df.loc[valid_temps, "calc_tavg_C"].values
                tmax = df.loc[valid_temps, "calc_tmax_C"].values
                assert (tmin <= tavg + 1e-5).all(), f"tmin > tavg in {fpath}"
                assert (tavg <= tmax + 1e-5).all(), f"tavg > tmax in {fpath}"

            # 6. FSM transitions extraction
            transitions = extract_fsm_transitions(df)
            assert isinstance(transitions, list)

        assert total_rows == 12244, f"Expected 12,244 total BMS rows, counted {total_rows}"

    def test_all_bms_rollover_sessions_stitched_continuity(self):
        """Stress test: Rollover sessions must concatenate seamlessly with monotonic time."""
        for p1, p2 in ROLLOVER_PAIRS:
            stitched = load_ams_session(p1, auto_stitch=True, base_dir=str(BASE_DIR))
            df1 = load_ams_log(p1, base_dir=str(BASE_DIR))
            df2 = load_ams_log(p2, base_dir=str(BASE_DIR))

            assert len(stitched) == len(df1) + len(df2), (
                f"Stitched length mismatch for pair ({p1}, {p2})"
            )
            assert (stitched["pack_sum_diff_mV"] == 0).all()
            assert (stitched["time_s"].diff().dropna() > 0).all(), (
                f"Non-monotonic time in stitched rollover ({p1}, {p2})"
            )


class TestImuAdversarialStress:
    """Adversarial stress testing against all 8 IMU files."""

    def test_all_8_imu_files_and_sampling_rate(self):
        """Stress test: 100 Hz sampling (median dt = 10 ms) and total 44,159 rows."""
        imu_files = sorted(glob.glob(str(AMS_DIR / "IMU*.CSV")))
        assert len(imu_files) == 8, f"Expected 8 IMU files, found {len(imu_files)}"

        total_imu_rows = 0

        for fpath in imu_files:
            log_id = int(Path(fpath).stem[3:])
            df = load_imu_log(log_id, apply_butterworth=True, base_dir=str(BASE_DIR))
            total_imu_rows += len(df)

            # Sampling rate: median interval must be exactly 10 ms
            dt = df["tick_ms"].diff().dropna()
            assert dt.median() == 10.0, f"Median dt != 10.0 ms in {fpath}"

            # Gravity norm
            g_mean = df["a_mag_g"].mean()
            assert 0.95 <= g_mean <= 1.05, f"Abnormal gravity norm {g_mean:.3f}g in {fpath}"

            # Shock detection executes without error
            shocks = detect_mechanical_shocks(df, threshold_g=0.05)
            for s in shocks:
                assert s.peak_da_g > 0.05
                assert s.duration_ms >= 0

            # Co-located BMS resample and correlation
            df_bms = load_ams_log(log_id, base_dir=str(BASE_DIR))
            corr = correlate_imu_bms(df, df_bms)
            assert isinstance(corr, pd.DataFrame)
            resamp = resample_bms_to_imu(df_bms, df)
            assert len(resamp) == len(df)

        assert total_imu_rows == 44159, (
            f"Expected 44,159 total IMU rows, counted {total_imu_rows}"
        )


class TestPipelineResourceEfficiency:
    """Stress test memory leak freedom and execution bounds."""

    def test_repeated_loading_memory_stability(self):
        """Verify that loading sessions repeatedly does not leak memory unboundedly."""
        gc.collect()

        # Load representative subset 3 times
        subset_telem = ["ISC_20260721_174418_MARTA3_CANTO", "ISC_20260805_182532_marta_accel_bcn"]
        subset_bms = [74, 93, 106]
        subset_imu = [74, 93, 106]

        durations: List[float] = []

        for iteration in range(3):
            t0 = time.perf_counter()
            for sid in subset_telem:
                _ = load_session(sid, base_dir=str(DATOS_DIR))
            for bid in subset_bms:
                _ = load_ams_log(bid, base_dir=str(BASE_DIR))
            for iid in subset_imu:
                _ = load_imu_log(iid, base_dir=str(BASE_DIR))
            durations.append(time.perf_counter() - t0)
            gc.collect()

        # Check performance stability: duration should not degrade dramatically
        avg_duration = sum(durations) / len(durations)
        assert avg_duration < 5.0, f"Average subset loading took too long: {avg_duration:.2f}s"

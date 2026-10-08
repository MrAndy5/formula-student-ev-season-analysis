"""Unit Test Suite for Formula Student EV Data Pipeline (Milestone 1).

Verifies cataloging, vehicle telemetry normalization, BMS parsing, and IMU dynamics.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import pytest

from src.pipeline.catalog import (
    build_season_catalog,
    compute_file_hash,
    detect_schema_id,
    export_catalog_artifacts,
    extract_provenance_tags,
    extract_session_metadata,
    get_active_sessions,
    get_stubs,
)
from src.pipeline.telemetry import (
    CANONICAL_COLUMNS,
    compute_calibrated_speed,
    detect_telemetry_schema,
    extract_standardized_time,
    load_session,
    load_telemetry_file,
    normalize_dcdc_current,
    normalize_pedals,
    normalize_telemetry_dataframe,
    resolve_telemetry_path,
)
from src.pipeline.bms import (
    DEAD_NTC_CHANNEL,
    FAULT_NAMES,
    FSM_NAMES,
    SENTINEL_OPEN,
    decode_detail,
    extract_fsm_transitions,
    filter_c1_artifact,
    load_ams_log,
    load_ams_session,
    resolve_bms_path,
)
from src.pipeline.imu import (
    correlate_imu_bms,
    detect_mechanical_shocks,
    load_imu_log,
    resample_bms_to_imu,
    resolve_imu_path,
)

BASE_DIR = Path(__file__).resolve().parent.parent


class TestCatalogUnit:
    """Unit tests for src/pipeline/catalog.py."""

    def test_catalog_census_count_and_columns(self):
        df = build_season_catalog(BASE_DIR)
        assert len(df) == 118
        expected_cols = [
            "session_id",
            "source_file",
            "source_dir",
            "schema_id",
            "row_count",
            "duration_s",
            "is_valid_run",
            "is_duplicate",
            "venue",
            "discipline",
            "driver",
            "date",
        ]
        assert list(df.columns) == expected_cols

    def test_catalog_stubs_and_duplicates(self):
        df = build_season_catalog(BASE_DIR)
        stubs = get_stubs(df)
        assert len(stubs) == 15
        assert (stubs["row_count"] == 0).all()
        assert (stubs["duration_s"] == 0.0).all()

        active = get_active_sessions(df)
        assert len(active) == 101  # 103 valid minus 2 duplicates
        dups = df[df["is_duplicate"] == True]
        assert len(dups) == 2

    def test_catalog_export_artifacts(self, tmp_path):
        df = build_season_catalog(BASE_DIR)
        csv_p, json_p = export_catalog_artifacts(
            df, artifacts_dir=tmp_path, root_dir=BASE_DIR
        )
        assert csv_p.exists()
        assert json_p.exists()
        assert os.path.getsize(csv_p) > 1000
        assert os.path.getsize(json_p) > 1000


class TestTelemetryUnit:
    """Unit tests for src/pipeline/telemetry.py."""

    def test_telemetry_empty_stub_graceful(self):
        # Pick one known stub
        stub_path = BASE_DIR / "datos2526" / "ISC_20260805_191320_prueba_box.csv"
        df = load_telemetry_file(stub_path, normalize=True)
        assert len(df) == 0
        for col in CANONICAL_COLUMNS:
            assert col in df.columns

    def test_telemetry_active_session_normalization(self):
        df = load_session("ISC_20260721_174418_MARTA3_CANTO")
        assert len(df) == 493
        for col in CANONICAL_COLUMNS:
            assert col in df.columns
        assert df["timestamp_s"].iloc[0] == 0.0
        assert (df["timestamp_s"].diff().dropna() >= 0.0).all()
        assert (df["speed_kmh"] >= 0.0).all()
        assert (df["i_dcdc_A"] >= 0.0).all()

    def test_telemetry_speed_calibration_formula(self):
        test_df = pd.DataFrame({"inv_rpm": [0.0, 3797.8, 7595.6]})
        speeds = compute_calibrated_speed(test_df)
        assert np.isclose(speeds.iloc[0], 0.0)
        assert np.isclose(speeds.iloc[1], 100.0)
        assert np.isclose(speeds.iloc[2], 200.0)

    def test_telemetry_dcdc_normalization_polarity(self):
        # July 19 (-380A -> +38A)
        july19_raw = pd.Series([-380.0, -370.0, -385.0])
        norm19 = normalize_dcdc_current(july19_raw)
        assert np.isclose(norm19.iloc[0], 38.0)
        assert (norm19 >= 0).all()

        # July 20 (-38A -> +38A)
        july20_raw = pd.Series([-38.0, -37.0, -38.5])
        norm20 = normalize_dcdc_current(july20_raw)
        assert np.isclose(norm20.iloc[0], 38.0)
        assert (norm20 >= 0).all()

        # July 21+ (+37A -> +37A)
        july21_raw = pd.Series([37.0, 38.0, 37.5])
        norm21 = normalize_dcdc_current(july21_raw)
        assert np.isclose(norm21.iloc[0], 37.0)

    def test_telemetry_ev_2_3_assertion(self):
        df = load_session(
            "ISC_20260730_092010_andres_dieguez_2_austria_endurance_2026_2 (1)"
        )
        assert (df["rule_ev_2_3_flag"] == 1).sum() >= 1


class TestBmsUnit:
    """Unit tests for src/pipeline/bms.py."""

    def test_bms_load_log_74_and_invariants(self):
        df = load_ams_log(74)
        assert len(df) == 69
        assert len(df.columns) >= 314
        assert (df["pack_sum_diff_mV"] == 0).all()
        assert df["t0_12"].isna().all()
        assert "t0_12_masked" in df.columns
        assert (df["time_s"].diff().dropna() > 0).all()

    def test_bms_module_offline_row_log93(self):
        df = load_ams_log(93)
        # Even with module 4 offline, pack_sum_diff_mV must be 0
        assert (df["pack_sum_diff_mV"] == 0).all()

    def test_bms_fsm_transitions_extraction(self):
        df = load_ams_log(74)
        transitions = extract_fsm_transitions(df)
        assert len(transitions) > 0
        # Check transition to ERR (5)
        err_trans = [t for t in transitions if t.to_state == 5]
        assert len(err_trans) == 1
        assert err_trans[0].fault_name == "NtcDisc"

    def test_bms_rollover_stitching(self):
        stitched = load_ams_session(89)
        assert len(stitched) == 1749  # 1200 from 89 + 549 from 90
        assert stitched["time_s"].iloc[0] == 0.0
        assert (stitched["time_s"].diff().dropna() > 0).all()


class TestImuUnit:
    """Unit tests for src/pipeline/imu.py."""

    def test_imu_load_74_and_filtering(self):
        df = load_imu_log(74)
        assert len(df) == 1724
        assert "da_mag_g" in df.columns
        assert "ax_filt_g" in df.columns
        assert (df["time_s"].diff().dropna() > 0).all()

    def test_imu_shock_detection(self):
        df = load_imu_log(74)
        shocks = detect_mechanical_shocks(df)
        assert len(shocks) > 0
        # Peak shock jerk should exceed 0.05g
        assert any(s.peak_da_g > 0.05 for s in shocks)

    def test_imu_bms_correlation(self):
        df_imu = load_imu_log(74)
        df_bms = load_ams_log(74)
        corr = correlate_imu_bms(df_imu, df_bms)
        assert len(corr) > 0
        assert "time_lag_ms" in corr.columns

    def test_resample_bms_to_imu(self):
        df_imu = load_imu_log(74)
        df_bms = load_ams_log(74)
        resampled = resample_bms_to_imu(df_bms, df_imu)
        assert len(resampled) == len(df_imu)
        assert "pack_V" in resampled.columns

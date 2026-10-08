"""Empirical Metric & Fuzzing Verification Suite (Milestone 1 - Challenger 2).

Test Suite: tests/test_m1_challenger_fuzzing.py
Author: M1 Challenger 2 (Empirical Metric & Fuzzing Verifier)
Mission:
1. Verify artifacts/season_session_catalog.csv (118 rows, 15 stubs with is_valid_run=False, 2 duplicates)
   and artifacts/session_summary.json validity against strict schema and disk reality.
2. Fuzzing test: test load_session and load_ams_log with invalid, nonexistent, or corrupted inputs.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import jsonschema
import numpy as np
import pandas as pd
import pytest

from src.pipeline.catalog import (
    CATALOG_COLUMNS,
    build_season_catalog,
    compute_file_hash,
)
from src.pipeline.telemetry import (
    CANONICAL_COLUMNS,
    load_session,
    load_telemetry_file,
    resolve_telemetry_path,
)
from src.pipeline.bms import (
    DEAD_NTC_CHANNEL,
    SENTINEL_OPEN,
    load_ams_log,
    resolve_bms_path,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_DIR = BASE_DIR / "datos2526"
AMS_DIR = BASE_DIR / "AMS logs" / "AMS logs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CATALOG_PATH = ARTIFACTS_DIR / "season_session_catalog.csv"
SUMMARY_PATH = ARTIFACTS_DIR / "session_summary.json"


# =============================================================================
# SUITE 1: Empirical Catalog & JSON Summary Verification
# =============================================================================

class TestCatalogAndSummaryVerification:
    """Rigorous empirical validation of the season catalog and summary artifacts."""

    def test_catalog_exact_118_census_vs_physical_disk(self):
        """Verify catalog contains exactly 118 rows matching the physical filesystem."""
        assert CATALOG_PATH.exists(), f"Catalog artifact missing at {CATALOG_PATH}"
        df = pd.read_csv(CATALOG_PATH)

        # 1. Total row count assertion
        assert len(df) == 118, f"Expected exactly 118 rows in catalog, found {len(df)}"

        # 2. Physical filesystem census
        telem_disk = sorted(DATOS_DIR.glob("*.csv"))
        bms_disk = sorted(AMS_DIR.glob("LOG*.CSV"))
        imu_disk = sorted(AMS_DIR.glob("IMU*.CSV"))

        assert len(telem_disk) == 78, f"Expected 78 telemetry files on disk, found {len(telem_disk)}"
        assert len(bms_disk) == 32, f"Expected 32 BMS LOG files on disk, found {len(bms_disk)}"
        assert len(imu_disk) == 8, f"Expected 8 IMU files on disk, found {len(imu_disk)}"
        total_disk = len(telem_disk) + len(bms_disk) + len(imu_disk)
        assert total_disk == 118, f"Expected 118 total files on disk, found {total_disk}"

        # 3. 1-to-1 bijection between catalog source_file and disk files
        catalog_files = set(df["source_file"])
        disk_files = {p.name for p in telem_disk + bms_disk + imu_disk}
        assert catalog_files == disk_files, (
            f"Catalog source_file set does not match physical files. "
            f"Missing in catalog: {disk_files - catalog_files}, Extra in catalog: {catalog_files - disk_files}"
        )

    def test_catalog_schema_columns_and_zero_nulls(self):
        """Verify all 12 contract columns exist and contain zero null / NaN entries."""
        df = pd.read_csv(CATALOG_PATH)
        assert list(df.columns) == CATALOG_COLUMNS, (
            f"Catalog columns mismatch.\nExpected: {CATALOG_COLUMNS}\nFound: {list(df.columns)}"
        )

        null_counts = df.isna().sum()
        for col, count in null_counts.items():
            assert count == 0, f"Catalog column '{col}' contains {count} null/NaN values."

    def test_catalog_stubs_exact_15_isolation_and_disk_proof(self):
        """Verify exactly 15 stubs with is_valid_run=False and duration_s=0.0."""
        df = pd.read_csv(CATALOG_PATH)
        stubs = df[df["is_valid_run"] == False]

        # 1. Exact count of stubs in catalog
        assert len(stubs) == 15, f"Expected exactly 15 stubs in catalog, got {len(stubs)}"
        assert (stubs["row_count"] == 0).all(), "Every stub must have row_count == 0"
        assert (stubs["duration_s"] == 0.0).all(), "Every stub must have duration_s == 0.0"
        assert (stubs["is_duplicate"] == False).all(), "Stubs must not be flagged as duplicates"
        assert (stubs["source_dir"] == "datos2526").all(), "All stubs reside in datos2526"

        # 2. Verify against physical files on disk
        for _, row in stubs.iterrows():
            fpath = DATOS_DIR / row["source_file"]
            assert fpath.exists(), f"Stub file does not exist on disk: {fpath}"
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                lines = [line.strip() for line in fp if line.strip()]
            assert len(lines) <= 1, f"Stub file {fpath.name} has {len(lines)} lines; expected <= 1"

        # 3. Verify non-stubs actually have > 0 rows
        active = df[df["is_valid_run"] == True]
        assert len(active) == 103, f"Expected 103 valid runs (118 - 15), found {len(active)}"
        assert (active["row_count"] > 0).all(), "All active valid runs must have row_count > 0"
        assert (active["duration_s"] > 0.0).all(), "All active valid runs must have duration_s > 0.0"

    def test_catalog_duplicates_exact_2_hash_collision_proof(self):
        """Verify exactly 2 duplicates with is_duplicate=True and bit-for-bit SHA256 match."""
        df = pd.read_csv(CATALOG_PATH)
        dups = df[df["is_duplicate"] == True]

        assert len(dups) == 2, f"Expected exactly 2 duplicate files, found {len(dups)}"
        assert (dups["is_valid_run"] == True).all(), "Duplicate files must be valid runs"

        dup_files = sorted(dups["source_file"].tolist())
        expected_dups = [
            "ISC_20260721_174418_MARTA3_CANTO (1).csv",
            "ISC_20260920_153953_Piloto_Test_Circuito_Test(1).csv",
        ]
        assert dup_files == sorted(expected_dups), f"Unexpected duplicate files: {dup_files}"

        # SHA-256 byte comparison with canonical counterparts
        pairs = [
            (
                DATOS_DIR / "ISC_20260721_174418_MARTA3_CANTO.csv",
                DATOS_DIR / "ISC_20260721_174418_MARTA3_CANTO (1).csv",
            ),
            (
                DATOS_DIR / "ISC_20260920_153953_Piloto_Test_Circuito_Test.csv",
                DATOS_DIR / "ISC_20260920_153953_Piloto_Test_Circuito_Test(1).csv",
            ),
        ]
        for canonical, duplicate in pairs:
            assert canonical.exists(), f"Canonical file missing: {canonical}"
            assert duplicate.exists(), f"Duplicate file missing: {duplicate}"
            h_canon = compute_file_hash(canonical)
            h_dup = compute_file_hash(duplicate)
            assert h_canon == h_dup, f"SHA-256 hash mismatch between {canonical.name} and {duplicate.name}"

    def test_session_summary_json_schema_validation(self):
        """Verify session_summary.json conforms strictly to expected JSON schema."""
        assert SUMMARY_PATH.exists(), f"Summary artifact missing at {SUMMARY_PATH}"
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary_schema = {
            "type": "object",
            "required": [
                "dataset_census",
                "subsystem_breakdown",
                "schema_distribution",
                "venue_distribution",
                "discipline_distribution",
                "driver_distribution",
                "stubs_inventory",
                "duplicate_pairs",
                "merged_sessions",
            ],
            "properties": {
                "dataset_census": {
                    "type": "object",
                    "required": [
                        "total_files",
                        "active_data_files",
                        "empty_stubs",
                        "total_telemetry_files",
                        "total_bms_files",
                        "total_imu_files",
                        "total_rows_logged",
                        "total_telemetry_rows",
                        "total_bms_rows",
                        "total_imu_rows",
                        "total_logged_duration_s",
                        "active_valid_duration_s",
                    ],
                    "properties": {
                        "total_files": {"type": "integer", "const": 118},
                        "active_data_files": {"type": "integer", "const": 103},
                        "empty_stubs": {"type": "integer", "const": 15},
                        "total_telemetry_files": {"type": "integer", "const": 78},
                        "total_bms_files": {"type": "integer", "const": 32},
                        "total_imu_files": {"type": "integer", "const": 8},
                        "total_rows_logged": {"type": "integer"},
                        "total_telemetry_rows": {"type": "integer"},
                        "total_bms_rows": {"type": "integer"},
                        "total_imu_rows": {"type": "integer"},
                        "total_logged_duration_s": {"type": "number"},
                        "active_valid_duration_s": {"type": "number"},
                    },
                },
                "subsystem_breakdown": {
                    "type": "object",
                    "required": ["telemetry", "bms", "imu"],
                },
                "schema_distribution": {"type": "object"},
                "venue_distribution": {"type": "object"},
                "discipline_distribution": {"type": "object"},
                "driver_distribution": {"type": "object"},
                "stubs_inventory": {
                    "type": "array",
                    "minItems": 15,
                    "maxItems": 15,
                    "items": {
                        "type": "object",
                        "required": ["source_file", "schema_id", "venue", "discipline", "driver", "date"],
                    },
                },
                "duplicate_pairs": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "required": ["canonical_file", "duplicate_file", "sha256", "row_count"],
                    },
                },
                "merged_sessions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["source_file", "session_id", "schema_id", "total_rows", "provenance_tags", "description"],
                    },
                },
            },
        }

        # jsonschema validation
        jsonschema.validate(instance=data, schema=summary_schema)

    def test_session_summary_mathematical_consistency_with_catalog(self):
        """Cross-check session_summary.json aggregates against catalog DataFrame values."""
        df = pd.read_csv(CATALOG_PATH)
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        census = data["dataset_census"]
        assert census["total_files"] == len(df)
        assert census["active_data_files"] == len(df[df["is_valid_run"] == True])
        assert census["empty_stubs"] == len(df[df["is_valid_run"] == False])
        assert census["total_telemetry_files"] == len(df[df["source_dir"] == "datos2526"])
        assert census["total_bms_files"] == len(df[df["schema_id"] == "schema_bms_314"])
        assert census["total_imu_files"] == len(df[df["schema_id"] == "schema_imu_7"])
        assert census["total_rows_logged"] == int(df["row_count"].sum())
        assert abs(census["total_logged_duration_s"] - round(float(df["duration_s"].sum()), 2)) < 0.05


# =============================================================================
# SUITE 2: Telemetry Ingestion Fuzzing & Robustness (load_session)
# =============================================================================

class TestTelemetryFuzzingAndRobustness:
    """Stress testing load_session with adversarial, malformed, and out-of-boundary inputs."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "nonexistent_session_12345",
            "NONEXISTENT_SESSION_RANDOM",
            99999999,
            "../../etc/passwd",
            "..\\..\\windows\\system32\\cmd.exe",
            "ISC_99999999_999999_no_file.csv",
        ],
    )
    def test_load_session_nonexistent_ids_raise_filenotfound(self, bad_id):
        """Verify load_session cleanly raises FileNotFoundError on non-existent targets."""
        with pytest.raises(FileNotFoundError):
            load_session(bad_id)

    @pytest.mark.parametrize("blank_id", ["", "   ", "\t\n"])
    def test_load_session_blank_ids_fail_cleanly(self, blank_id):
        """Verify load_session raises an exception on empty or blank strings without hanging."""
        with pytest.raises((FileNotFoundError, PermissionError, IsADirectoryError, ValueError)):
            load_session(blank_id)

    def test_load_session_none_fails_cleanly(self):
        """Verify load_session(None) raises an exception gracefully."""
        with pytest.raises((FileNotFoundError, TypeError)):
            load_session(None)

    def test_load_session_empty_stubs_adhere_to_canonical_schema(self):
        """Empirically test all 15 stubs through load_session to guarantee contract compliance."""
        df_cat = pd.read_csv(CATALOG_PATH)
        stubs = df_cat[df_cat["is_valid_run"] == False]

        for _, stub in stubs.iterrows():
            session_id = stub["session_id"]
            df_loaded = load_session(session_id)

            assert isinstance(df_loaded, pd.DataFrame)
            assert len(df_loaded) == 0, f"Stub {session_id} should return 0 rows"
            for col in CANONICAL_COLUMNS:
                assert col in df_loaded.columns, f"Canonical column '{col}' missing from stub load of {session_id}"

    def test_load_session_corrupted_payloads_fuzzing(self, tmp_path):
        """Fuzz load_telemetry_file with synthesized corrupted files."""
        # 1. 0-byte file
        f_0byte = tmp_path / "zero_byte.csv"
        f_0byte.write_bytes(b"")
        df_0 = load_telemetry_file(str(f_0byte))
        assert isinstance(df_0, pd.DataFrame)
        assert len(df_0) == 0
        for col in CANONICAL_COLUMNS:
            assert col in df_0.columns

        # 2. Whitespace-only file
        f_ws = tmp_path / "whitespace.csv"
        f_ws.write_text("   \n\t  \n   \n", encoding="utf-8")
        df_ws = load_telemetry_file(str(f_ws))
        assert isinstance(df_ws, pd.DataFrame)
        assert len(df_ws) == 0
        for col in CANONICAL_COLUMNS:
            assert col in df_ws.columns

        # 3. Ragged CSV lines (inconsistent field counts)
        f_ragged = tmp_path / "ragged.csv"
        f_ragged.write_text("col_a,col_b,col_c\n1,2\n3,4,5,6,7,8\n", encoding="utf-8")
        with pytest.raises((pd.errors.ParserError, ValueError)):
            load_telemetry_file(str(f_ragged))

        # 4. Binary fuzzing junk (random byte sequences)
        for seed in [42, 1337, 9999]:
            np.random.seed(seed)
            f_bin = tmp_path / f"bin_junk_{seed}.csv"
            # Random non-ASCII bytes
            f_bin.write_bytes(np.random.bytes(1024))
            with pytest.raises((UnicodeDecodeError, pd.errors.ParserError, Exception)):
                load_telemetry_file(str(f_bin))

        # 5. String tokens in numeric channels
        f_strnum = tmp_path / "strnum.csv"
        f_strnum.write_text(
            "inv_rpm,inv_dc_bus_V,time\nNaN,100.0,2026-07-21 17:00:00\nMALFORMED_STRING,500.0,2026-07-21 17:00:01\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_telemetry_file(str(f_strnum))

        # 6. Unknown schema layout (graceful fallback)
        f_unknown = tmp_path / "unknown_schema.csv"
        f_unknown.write_text(
            "time,random_sensor_1,random_sensor_2\n2026-07-21 17:00:00,123.4,567.8\n2026-07-21 17:00:01,234.5,678.9\n",
            encoding="utf-8",
        )
        df_unk = load_telemetry_file(str(f_unknown))
        assert isinstance(df_unk, pd.DataFrame)
        assert len(df_unk) == 2
        for col in CANONICAL_COLUMNS:
            assert col in df_unk.columns


# =============================================================================
# SUITE 3: BMS Ingestion Fuzzing & Invariant Robustness (load_ams_log)
# =============================================================================

class TestBmsFuzzingAndRobustness:
    """Stress testing load_ams_log with adversarial, malformed, and corrupted inputs."""

    @pytest.mark.parametrize(
        "bad_log_id",
        [
            99999,
            -1,
            -999,
            "LOG9999",
            "NONEXISTENT_BMS",
            "../../etc/passwd",
            "LOG0000",
        ],
    )
    def test_load_ams_log_nonexistent_ids_raise_filenotfound(self, bad_log_id):
        """Verify load_ams_log cleanly raises FileNotFoundError on missing files."""
        with pytest.raises(FileNotFoundError):
            load_ams_log(bad_log_id)

    @pytest.mark.parametrize("blank_id", ["", "   "])
    def test_load_ams_log_blank_ids_fail_cleanly(self, blank_id):
        """Verify load_ams_log raises an exception on empty or blank inputs."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_ams_log(blank_id)

    def test_load_ams_log_none_fails_cleanly(self):
        """Verify load_ams_log(None) raises TypeError gracefully."""
        with pytest.raises((TypeError, FileNotFoundError)):
            load_ams_log(None)

    def test_load_ams_log_corrupted_payloads_fuzzing(self, tmp_path):
        """Fuzz load_ams_log with corrupted files."""
        # 1. 0-byte file
        f_0 = tmp_path / "LOG9901.CSV"
        f_0.write_bytes(b"")
        with pytest.raises((pd.errors.EmptyDataError, ValueError, KeyError)):
            load_ams_log(str(f_0))

        # 2. Whitespace file
        f_ws = tmp_path / "LOG9902.CSV"
        f_ws.write_text("    \n\t   \n", encoding="utf-8")
        with pytest.raises((pd.errors.EmptyDataError, ValueError, KeyError)):
            load_ams_log(str(f_ws))

        # 3. Random binary bytes
        f_bin = tmp_path / "LOG9903.CSV"
        f_bin.write_bytes(np.random.bytes(1024))
        with pytest.raises((UnicodeDecodeError, pd.errors.ParserError, Exception)):
            load_ams_log(str(f_bin))

        # 4. Incomplete/missing BMS columns
        f_partial = tmp_path / "LOG9904.CSV"
        f_partial.write_text("tick_ms,pack_mV\n100,380000\n200,379900\n", encoding="utf-8")
        with pytest.raises(KeyError):
            # Missing I_raw_mA, I_filt_mA, Idcdc_mA, mod_mask, etc.
            load_ams_log(str(f_partial))

    def test_load_ams_log_mathematical_pack_voltage_invariant_on_physical_data(self):
        """Verify pack_mV bitmask invariant across a representative sample of BMS logs."""
        # Test on LOG0074 (standard bench run) and LOG0093 (Module 4 offline)
        for log_id in [74, 93]:
            df = load_ams_log(log_id)
            assert "pack_sum_diff_mV" in df.columns
            # Pack voltage bitmask sum must have exact 0.0 mV residual
            max_diff = df["pack_sum_diff_mV"].abs().max()
            assert max_diff == 0.0, f"Pack voltage bitmask invariant violated in LOG{log_id:04d}: max diff = {max_diff} mV"

    def test_load_ams_log_thermal_sentinel_masking_adherence(self):
        """Verify that -32768 sentinels are completely masked to NaN and t0_12 is masked."""
        df = load_ams_log(74, mask_sentinels=True, mask_dead_channel=True)
        temp_cols = [f"t{m}_{n}" for m in range(5) for n in range(40)]

        # No open-circuit sentinels remaining
        for c in temp_cols:
            assert not (df[c] == SENTINEL_OPEN).any(), f"Sentinel {SENTINEL_OPEN} found in column {c}"

        # Dead channel t0_12 must be 100% NaN
        assert df[DEAD_NTC_CHANNEL].isna().all(), f"Dead channel {DEAD_NTC_CHANNEL} was not fully masked"
        assert df["t0_12_masked"].iloc[0] is True or df["t0_12_masked"].iloc[0] == 1

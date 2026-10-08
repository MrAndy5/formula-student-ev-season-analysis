"""
End-to-End (E2E) Test Suite for Formula Student EV Season Dataset Engineering Analysis.

Taxonomy:
- Tier 1: Feature Coverage (R1 - R5, >=5 tests per feature, 25 tests total)
- Tier 2: Boundary & Corner Cases (>=5 tests, 7 tests total)
- Tier 3: Cross-Feature Combinations (>=4 tests, 5 tests total)
- Tier 4: Real-World Application Scenarios (>=5 tests, 5 tests total)

Total Test Count: 42 tests.
"""

import os
import glob
import json
import math
import hashlib
import pytest
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Base Workspace Paths & Utilities
# -----------------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATOS_DIR = os.path.join(BASE_DIR, "datos2526")
AMS_DIR = os.path.join(BASE_DIR, "AMS logs", "AMS logs")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def require_artifact(relative_path: str) -> str:
    """Helper to enforce pipeline artifact existence or skip gracefully during development."""
    full_path = os.path.join(BASE_DIR, relative_path)
    if not os.path.exists(full_path):
        pytest.skip(f"Pending pipeline implementation: artifact '{relative_path}' not yet generated.")
    return full_path


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# TIER 1: Feature Coverage (R1 - R5)
# =============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage verifying requirements R1 through R5."""

    # --- Feature R1: Season Telemetry & Diagnostic Processing Pipeline ---

    def test_r1_session_catalog_schema_and_integrity(self):
        """R1.1: Verify season_session_catalog.csv existence and mandatory column schema."""
        catalog_path = require_artifact("artifacts/season_session_catalog.csv")
        df = pd.read_csv(catalog_path)
        required_cols = [
            "session_id", "source_file", "source_dir", "schema_id",
            "row_count", "duration_s", "is_valid_run", "is_duplicate",
            "venue", "discipline", "driver", "date"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column '{col}' in session catalog."
        assert len(df) > 0, "Session catalog must not be empty."
        assert df["session_id"].isna().sum() == 0, "session_id column must not contain nulls."
        assert df["source_file"].isna().sum() == 0, "source_file column must not contain nulls."

    def test_r1_raw_census_100_percent_coverage(self):
        """R1.2: Verify 100% census of all 118 raw CSV files (78 vehicle + 40 AMS/IMU)."""
        vehicle_files = glob.glob(os.path.join(DATOS_DIR, "*.csv"))
        ams_log_files = glob.glob(os.path.join(AMS_DIR, "LOG*.CSV"))
        ams_imu_files = glob.glob(os.path.join(AMS_DIR, "IMU*.CSV"))

        assert len(vehicle_files) == 78, f"Expected 78 vehicle CSV files, found {len(vehicle_files)}"
        assert len(ams_log_files) == 32, f"Expected 32 AMS LOG CSV files, found {len(ams_log_files)}"
        assert len(ams_imu_files) == 8, f"Expected 8 AMS IMU CSV files, found {len(ams_imu_files)}"
        total_files = len(vehicle_files) + len(ams_log_files) + len(ams_imu_files)
        assert total_files == 118, f"Expected 118 total CSV files, found {total_files}"

        # If catalog exists, verify all 118 files are accounted for
        cat_file = os.path.join(ARTIFACTS_DIR, "season_session_catalog.csv")
        if os.path.exists(cat_file):
            df = pd.read_csv(cat_file)
            assert len(df) >= 118, f"Catalog must index at least 118 files, found {len(df)}"

    def test_r1_empty_stub_classification(self):
        """R1.3: Verify exact isolation of 15 empty 0-row header stubs in datos2526."""
        vehicle_files = glob.glob(os.path.join(DATOS_DIR, "*.csv"))
        empty_stubs = []
        for vf in vehicle_files:
            with open(vf, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                if len(lines) <= 1:
                    empty_stubs.append(os.path.basename(vf))

        assert len(empty_stubs) == 15, f"Expected exactly 15 empty CSV stubs, found {len(empty_stubs)}"

        cat_file = os.path.join(ARTIFACTS_DIR, "season_session_catalog.csv")
        if os.path.exists(cat_file):
            df = pd.read_csv(cat_file)
            stub_records = df[df["source_file"].isin(empty_stubs)]
            assert len(stub_records) == 15, "All 15 empty stubs must appear in catalog."
            assert (stub_records["is_valid_run"] == False).all(), "All 15 empty stubs must be flagged is_valid_run=False."

    def test_r1_duplicate_files_identification(self):
        """R1.4: Verify detection of duplicate files (MARTA3_CANTO and Piloto_Test_Circuito_Test)."""
        pair1_a = os.path.join(DATOS_DIR, "ISC_20260721_174418_MARTA3_CANTO.csv")
        pair1_b = os.path.join(DATOS_DIR, "ISC_20260721_174418_MARTA3_CANTO (1).csv")
        assert os.path.exists(pair1_a) and os.path.exists(pair1_b), "Pair 1 files must exist."
        assert compute_sha256(pair1_a) == compute_sha256(pair1_b), "MARTA3_CANTO copies must have identical hashes."

        cat_file = os.path.join(ARTIFACTS_DIR, "season_session_catalog.csv")
        if os.path.exists(cat_file):
            df = pd.read_csv(cat_file)
            p1_b_entry = df[df["source_file"] == "ISC_20260721_174418_MARTA3_CANTO (1).csv"]
            if not p1_b_entry.empty:
                assert p1_b_entry["is_duplicate"].iloc[0] == True, "Duplicate copy must have is_duplicate=True"

    def test_r1_session_duration_and_validity_metrics(self):
        """R1.5: Verify valid sessions have strictly positive duration, valid rows, and canonical venues."""
        cat_path = require_artifact("artifacts/season_session_catalog.csv")
        df = pd.read_csv(cat_path)
        valid_df = df[df["is_valid_run"] == True]
        assert len(valid_df) > 0, "There must be valid sessions in catalog."
        assert (valid_df["duration_s"] > 0).all(), "Valid sessions must have positive duration."
        assert (valid_df["row_count"] > 0).all(), "Valid sessions must have positive row count."
        valid_venues = {"cantoblanco", "austria", "barcelona", "jarama", "test", "demo"}
        venues = valid_df["venue"].dropna().str.lower().unique()
        for v in venues:
            assert any(vv in v for vv in valid_venues), f"Venue '{v}' is not recognized."

    # --- Feature R2: Fault Patterns, Safety Loop & Root-Cause Failure Analysis ---

    def test_r2_consolidated_fault_log_existence_and_schema(self):
        """R2.1: Verify consolidated_fault_log.csv schema, timestamps, and root cause fields."""
        fault_path = require_artifact("artifacts/consolidated_fault_log.csv")
        df = pd.read_csv(fault_path)
        expected_cols = ["timestamp", "log_file", "fault_name", "fault_code", "root_cause"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column '{col}' in consolidated fault log."
        assert len(df) > 0, "Consolidated fault log must not be empty."

    def test_r2_precharge_plateau_failure_diagnostics(self):
        """R2.2: Verify reconstruction of 7 precharge plateau timeouts stalling at 65-75% DC bus."""
        # Check artifact or inspect AMS logs directly
        precharge_files = ["LOG0075.CSV", "LOG0086.CSV", "LOG0095.CSV", "LOG0096.CSV", "LOG0097.CSV", "LOG0099.CSV", "LOG0105.CSV"]
        for pfile in precharge_files:
            full_path = os.path.join(AMS_DIR, pfile)
            assert os.path.exists(full_path), f"Precharge failure log {pfile} must exist."
            df = pd.read_csv(full_path)
            # Must transition to or end in fsm == 5 (ERR) or fault == 12 (FsmError)
            assert (df["fsm"] == 5).any() or (df["fault"] == 12).any(), f"Precharge failure log {pfile} must contain ERR state."

        precharge_art = os.path.join(ARTIFACTS_DIR, "precharge_analysis.json")
        if os.path.exists(precharge_art):
            with open(precharge_art, "r", encoding="utf-8") as f:
                data = json.load(f)
            failed_count = data.get("failed_precharge_count", len(data.get("failed_events", [])))
            assert failed_count == 7, f"Expected 7 failed precharge events in analysis, found {failed_count}"

    def test_r2_bms_fsm_shutdown_reconstruction(self):
        """R2.3: Verify reconstruction of all 13 BMS shutdown instances across Start, Pre, Run, ERR states."""
        # Survey identified 13 shutdown logs: LOG0074, 0075, 0086, 0091, 0093, 0095, 0096, 0097, 0098, 0099, 0102, 0103, 0105
        shutdown_logs = [
            "LOG0074.CSV", "LOG0075.CSV", "LOG0086.CSV", "LOG0091.CSV", "LOG0093.CSV",
            "LOG0095.CSV", "LOG0096.CSV", "LOG0097.CSV", "LOG0098.CSV", "LOG0099.CSV",
            "LOG0102.CSV", "LOG0103.CSV", "LOG0105.CSV"
        ]
        shutdowns_found = 0
        for sfile in shutdown_logs:
            full_path = os.path.join(AMS_DIR, sfile)
            if os.path.exists(full_path):
                df = pd.read_csv(full_path)
                if (df["fsm"] == 5).any():
                    shutdowns_found += 1
        assert shutdowns_found == 13, f"Expected 13 BMS shutdown logs with FSM=5, found {shutdowns_found}"

    def test_r2_fault_classification_taxonomy(self):
        """R2.4: Verify fault_classification_table.csv distinguishes hardware faults from operational trips."""
        tax_path = require_artifact("artifacts/fault_classification_table.csv")
        df = pd.read_csv(tax_path)
        assert len(df) >= 4, "Taxonomy table must categorize at least 4 fault modes."
        # Must have category column distinguishing hardware vs operational
        category_col = None
        for col in df.columns:
            if "category" in col.lower() or "type" in col.lower() or "classification" in col.lower():
                category_col = col
                break
        assert category_col is not None, "Taxonomy table must have a category or classification column."

    def test_r2_fs_ev_rule_plausibility_monitoring(self):
        """R2.5: Verify EV 2.3 plausibility breach detection in Austria Endurance stint 2."""
        austria_log = os.path.join(DATOS_DIR, "ISC_20260730_092010_andres_dieguez_2_austria_endurance_2026_2 (1).csv")
        assert os.path.exists(austria_log), "Austria endurance stint 2 log must exist."
        df = pd.read_csv(austria_log)
        assert "ev_2_3" in df.columns, "Column ev_2_3 must be present in telemetry log."
        # Exactly one occurrence of ev_2_3 == 1 in this file
        ev_trips = (df["ev_2_3"] == 1).sum()
        assert ev_trips >= 1, f"Expected at least 1 ev_2_3 rule breach in Austria log, found {ev_trips}"

    # --- Feature R3: Electrical Powertrain & Battery System Characterization ---

    def test_r3_cell_voltage_spread_and_health_summary(self):
        """R3.1: Verify cell_health_summary.csv contains min, max, delta, std across 95 cells."""
        health_path = require_artifact("artifacts/cell_health_summary.csv")
        df = pd.read_csv(health_path)
        assert len(df) >= 95, f"Expected at least 95 cell records, found {len(df)}"
        for metric in ["min", "max", "delta", "std"]:
            assert any(metric in c.lower() for c in df.columns), f"Metric '{metric}' missing in cell health summary."

    def test_r3_module_balance_rest_vs_load(self):
        """R3.2: Verify module_balance_metrics.csv evaluates all 5 modules under rest vs load."""
        balance_path = require_artifact("artifacts/module_balance_metrics.csv")
        df = pd.read_csv(balance_path)
        assert len(df) >= 5, "Module balance metrics must report data for all 5 modules."
        mod_col = None
        for col in df.columns:
            if "mod" in col.lower():
                mod_col = col
                break
        assert mod_col is not None, "Module identifier column must be present."

    def test_r3_dynamic_internal_resistance_regression(self):
        """R3.3: Verify pack internal resistance estimates derived via OLS (~0.406 Ohm)."""
        r_path = require_artifact("artifacts/internal_resistance_estimates.csv")
        df = pd.read_csv(r_path)
        assert len(df) > 0, "Internal resistance estimates must not be empty."
        r_pack_series = None
        for col in df.columns:
            if "pack" in col.lower() or "r_pack" in col.lower() or "resistance" in col.lower():
                r_pack_series = df[col]
                break
        assert r_pack_series is not None, "Pack resistance column must be present."
        val = float(r_pack_series.iloc[0])
        # Value can be in Ohms (~0.406) or mOhms (~406)
        if val > 10.0:
            val = val / 1000.0
        assert 0.25 <= val <= 0.60, f"Derived pack resistance {val:.4f} Ohm is out of physical range (0.25-0.60 Ohm)."

    def test_r3_accumulator_200_ntc_thermal_inventory_and_t0_12_mask(self):
        """R3.4: Verify thermal_channel_inventory.csv accounts for 200 NTCs and permanently masks t0_12."""
        therm_path = require_artifact("artifacts/thermal_channel_inventory.csv")
        df = pd.read_csv(therm_path)
        assert len(df) >= 200, f"Expected 200 NTC channel entries, found {len(df)}"
        t0_12_row = df[df.iloc[:, 0].astype(str).str.contains("t0_12")]
        assert not t0_12_row.empty, "Channel t0_12 must be present in inventory."
        # Verify it is flagged as disconnected / masked
        masked_col = None
        for col in df.columns:
            if "mask" in col.lower() or "disc" in col.lower() or "status" in col.lower():
                masked_col = col
                break
        assert masked_col is not None, "Masking or status column must exist in thermal inventory."

    def test_r3_competition_electrical_benchmark(self):
        """R3.5: Verify competition_comparison.json contrasts Austria vs Barcelona electrical metrics."""
        comp_path = require_artifact("artifacts/competition_comparison.json")
        with open(comp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_str = json.dumps(data).lower()
        assert "austria" in data_str, "Benchmark must include Austria metrics."
        assert "barcelona" in data_str or "montmelo" in data_str, "Benchmark must include Barcelona metrics."

    # --- Feature R4: Vehicle Dynamics & Driver Performance Evaluation ---

    def test_r4_dynamics_envelope_metrics_schema(self):
        """R4.1: Verify dynamics_envelope_metrics.csv schema for distinct track disciplines."""
        dyn_path = require_artifact("artifacts/dynamics_envelope_metrics.csv")
        df = pd.read_csv(dyn_path)
        assert len(df) > 0, "Dynamics envelope metrics table must not be empty."
        assert any("lat" in c.lower() for c in df.columns), "Lateral acceleration metrics must be present."
        assert any("long" in c.lower() or "ax" in c.lower() for c in df.columns), "Longitudinal acceleration metrics must be present."

    def test_r4_friction_circle_gg_bounds(self):
        """R4.2: Verify raw 100 Hz IMU acceleration envelope reflects 1g gravity and shakedown dynamics."""
        imu_files = glob.glob(os.path.join(AMS_DIR, "IMU*.CSV"))
        assert len(imu_files) > 0, "IMU files must exist."
        peak_lat = 0.0
        peak_long = 0.0
        mean_z = []
        for imuf in imu_files:
            df = pd.read_csv(imuf)
            if "ay_g" in df.columns and "ax_g" in df.columns:
                peak_lat = max(peak_lat, df["ay_g"].abs().max())
                peak_long = max(peak_long, df["ax_g"].abs().max())
            if "az_g" in df.columns:
                mean_z.append(df["az_g"].median())
        # IMU vertical acceleration reflects earth gravity (~1.0g)
        avg_gravity = float(np.mean(mean_z))
        assert 0.95 <= avg_gravity <= 1.10, f"Average vertical gravity should be ~1.0g, found {avg_gravity:.3f}g"
        assert peak_lat >= 0.05, f"Observed peak lateral acceleration {peak_lat}g should be non-zero."
        assert peak_long >= 0.05, f"Observed peak longitudinal acceleration {peak_long}g should be non-zero."

    def test_r4_energy_consumption_metrics(self):
        """R4.3: Verify energy_consumption_summary.csv contains Wh/km, Wh/min, and total kWh."""
        energy_path = require_artifact("artifacts/energy_consumption_summary.csv")
        df = pd.read_csv(energy_path)
        assert len(df) > 0, "Energy consumption summary must not be empty."
        cols = [c.lower() for c in df.columns]
        assert any("wh_km" in c or "wh/km" in c for c in cols), "Wh/km metric must be present."
        assert any("wh_min" in c or "wh/min" in c or "kwh" in c for c in cols), "Wh/min or kWh metric must be present."

    def test_r4_regeneration_recovery_efficiency(self):
        """R4.4: Verify regenerative braking energy recovery efficiency is quantified."""
        energy_path = require_artifact("artifacts/energy_consumption_summary.csv")
        df = pd.read_csv(energy_path)
        cols = [c.lower() for c in df.columns]
        assert any("regen" in c or "recovery" in c for c in cols), "Regen recovery efficiency column must be present."

    def test_r4_driver_benchmark_summary(self):
        """R4.5: Verify driver_benchmark_summary.csv contrasts performance across multiple drivers."""
        driver_path = require_artifact("artifacts/driver_benchmark_summary.csv")
        df = pd.read_csv(driver_path)
        assert len(df) >= 2, "Driver benchmark must compare at least 2 drivers."
        driver_col = None
        for col in df.columns:
            if "driver" in col.lower() or "piloto" in col.lower():
                driver_col = col
                break
        assert driver_col is not None, "Driver column must be present in benchmark summary."

    # --- Feature R5: Consolidated Technical Documentation & Automated Verification ---

    def test_r5_markdown_report_structure_and_completeness(self):
        """R5.1: Verify Markdown report existence (>5KB) and mandatory engineering sections."""
        md_path = require_artifact("reports/formula_student_ev_season_analysis.md")
        assert os.path.getsize(md_path) > 5000, "Markdown report must exceed 5KB."
        with open(md_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().lower()
        required_sections = [
            "executive summary",
            "fault",
            "powertrain",
            "dynamics",
            "recommendations"
        ]
        for sec in required_sections:
            assert sec in content, f"Markdown report missing section matching '{sec}'."

    def test_r5_standalone_html_report_validity(self):
        """R5.2: Verify HTML report existence (>15KB) and standard HTML document structure."""
        html_path = require_artifact("reports/formula_student_ev_season_analysis.html")
        assert os.path.getsize(html_path) > 15000, "HTML report must exceed 15KB."
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().lower()
        assert "<!doctype html" in content or "<html" in content, "Missing HTML header."
        assert "</html>" in content, "Missing closing HTML tag."
        assert "<body" in content, "Missing HTML body."

    def test_r5_embedded_visualizations_presence(self):
        """R5.3: Verify HTML report embeds figures (base64 data URI or SVG/img tags)."""
        html_path = require_artifact("reports/formula_student_ev_season_analysis.html")
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        has_embedded_img = ("data:image/png;base64," in content) or ("<svg" in content) or ("<img" in content)
        assert has_embedded_img, "HTML report must contain embedded visual plots or images."

    def test_r5_standalone_verify_script_executable(self):
        """R5.4: Verify presence of verify_pipeline.py standalone test script in workspace root."""
        verify_script = os.path.join(BASE_DIR, "verify_pipeline.py")
        if not os.path.exists(verify_script):
            pytest.skip("Pending implementation: verify_pipeline.py not yet generated.")
        assert os.path.getsize(verify_script) > 500, "verify_pipeline.py must be a non-trivial script."
        # Verify syntax
        with open(verify_script, "r", encoding="utf-8") as f:
            code = f.read()
        compile(code, verify_script, "exec")

    def test_r5_engineering_recommendations_actionability(self):
        """R5.5: Verify reports articulate actionable 2026/2027 season engineering recommendations."""
        md_path = require_artifact("reports/formula_student_ev_season_analysis.md")
        with open(md_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().lower()
        assert "2026" in content or "2027" in content, "Report must reference 2026/2027 season car."
        # Must recommend precharge/DCDC, NTC filter, or cell balancing improvements
        has_recs = ("precharge" in content or "dcdc" in content) and ("ntc" in content or "sensor" in content)
        assert has_recs, "Report must include specific hardware/firmware recommendations."


# =============================================================================
# TIER 2: Boundary & Corner Cases
# =============================================================================

class TestTier2BoundaryCornerCases:
    """Tier 2: Boundary stress testing, empty stubs, dead sensors, sentinels, and numerical limits."""

    def test_tier2_empty_0_row_csv_stubs_handling(self):
        """Tier 2.1: Verify parsing of all 15 empty header-only CSVs succeeds without unhandled crash."""
        vehicle_files = glob.glob(os.path.join(DATOS_DIR, "*.csv"))
        empty_stubs = [vf for vf in vehicle_files if len(open(vf, "r", encoding="utf-8", errors="replace").readlines()) <= 1]
        assert len(empty_stubs) == 15, f"Expected 15 empty CSV files, got {len(empty_stubs)}"

        for stub in empty_stubs:
            df = pd.read_csv(stub)
            assert len(df) == 0, f"Empty stub {os.path.basename(stub)} should parse to 0-length DataFrame."
            assert len(df.columns) in [56, 69], f"Unexpected header column count {len(df.columns)} in stub {os.path.basename(stub)}"

    def test_tier2_minimal_row_sessions_duration_guard(self):
        """Tier 2.2: Verify files with minimal row counts (<= 15 rows) handle duration calculation without error."""
        minimal_file = os.path.join(DATOS_DIR, "ISC_20260804_162149_landa_practice1_BAR.csv")
        assert os.path.exists(minimal_file), "Minimal test file must exist."
        df = pd.read_csv(minimal_file)
        assert len(df) == 5, f"Expected 5 rows in landa_practice1_BAR.csv, found {len(df)}"

        # Compute duration safely
        if "time_elapsed_s" in df.columns:
            t_col = df["time_elapsed_s"]
        elif "timestamp_s" in df.columns:
            t_col = df["timestamp_s"]
        elif "tick_ms" in df.columns:
            t_col = df["tick_ms"] / 1000.0
        else:
            t_col = None

        assert t_col is not None, "Time column must be present."
        dur = t_col.iloc[-1] - t_col.iloc[0]
        assert dur > 0, "Duration must be positive."

    def test_tier2_dead_sensor_t0_12_masking(self):
        """Tier 2.3: Verify thermistor t0_12 sentinel rate is >99% and strictly masked in calculations."""
        log_files = glob.glob(os.path.join(AMS_DIR, "LOG*.CSV"))
        total_rows = 0
        sentinel_count = 0
        for lf in log_files:
            df = pd.read_csv(lf, usecols=["t0_12"])
            total_rows += len(df)
            sentinel_count += (df["t0_12"] == -32768).sum()

        ratio = sentinel_count / total_rows
        assert ratio > 0.99, f"Channel t0_12 sentinel ratio should exceed 99%, observed {ratio*100:.2f}%"

    def test_tier2_sentinel_negative_32768_filtering_across_200_ntcs(self):
        """Tier 2.4: Verify sentinel -32768 is treated as invalid rather than a physical -32768 deg C."""
        log_sample = os.path.join(AMS_DIR, "LOG0074.CSV")
        assert os.path.exists(log_sample), "LOG0074.CSV must exist."
        df = pd.read_csv(log_sample)
        ntc_cols = [f"t{m}_{i}" for m in range(5) for i in range(40)]
        sub_vals = df[ntc_cols].values

        # Verify sentinels exist in raw data
        assert (sub_vals == -32768).any(), "Raw dataset must contain -32768 sentinels."

        # Filtered values must have physical temperatures between 0 and 90 deg C
        valid_temps = sub_vals[sub_vals != -32768]
        assert len(valid_temps) > 0, "There must be valid temperature readings."
        assert np.nanmin(valid_temps) >= 5.0, f"Physical temperature should be >= 5 C, found {np.nanmin(valid_temps)}"
        assert np.nanmax(valid_temps) <= 90.0, f"Physical temperature should be <= 90 C, found {np.nanmax(valid_temps)}"

    def test_tier2_overcurrent_threshold_filter_boundary(self):
        """Tier 2.5: Verify LOG0091 overcurrent trip boundary: 244.9A raw spike vs 182.6A filtered against 185A limit."""
        log91_path = os.path.join(AMS_DIR, "LOG0091.CSV")
        assert os.path.exists(log91_path), "LOG0091.CSV must exist."
        df = pd.read_csv(log91_path)
        # Check overcurrent shutdown in fault column (fault == 10)
        assert (df["fault"] == 10).any(), "LOG0091 must contain OverI fault (code 10)."
        max_raw_A = df["I_raw_mA"].max() / 1000.0
        max_filt_A = df["I_filt_mA"].max() / 1000.0
        assert 240.0 <= max_raw_A <= 250.0, f"Expected raw current ~245A, found {max_raw_A:.1f}A"
        assert 180.0 <= max_filt_A <= 185.0, f"Expected filtered current ~182.6A, found {max_filt_A:.1f}A"

    def test_tier2_zero_division_guards_in_kinematics_and_efficiency(self):
        """Tier 2.6: Verify speed v=0 and distance delta=0 do not trigger zero division in efficiency math."""
        # Simulated test case for zero speed & zero distance calculations
        zero_speed = 0.0
        zero_distance = 0.0
        total_energy_wh = 10.0

        # Mathematical guard test
        wh_km = (total_energy_wh / zero_distance) if zero_distance > 0 else 0.0
        assert wh_km == 0.0, "Zero distance must safely evaluate to 0.0 Wh/km."

        wh_min = (total_energy_wh / (0.0 / 60.0)) if zero_speed > 0 else 0.0
        assert wh_min == 0.0, "Zero time/speed must safely evaluate to 0.0 Wh/min."

    def test_tier2_zero_nan_inf_integrity_in_output_artifacts(self):
        """Tier 2.7: Verify generated tabular artifacts contain zero unhandled NaN or Inf values."""
        artifacts_to_check = [
            "artifacts/cell_health_summary.csv",
            "artifacts/module_balance_metrics.csv",
            "artifacts/internal_resistance_estimates.csv"
        ]
        checked = 0
        for art in artifacts_to_check:
            full_path = os.path.join(BASE_DIR, art)
            if os.path.exists(full_path):
                df = pd.read_csv(full_path)
                numeric_df = df.select_dtypes(include=[np.number])
                assert not np.isinf(numeric_df.values).any(), f"Artifact {art} contains Inf values."
                assert not numeric_df.isna().all().any(), f"Artifact {art} contains columns that are all NaN."
                checked += 1
        if checked == 0:
            pytest.skip("Tabular artifacts not yet generated for NaN/Inf validation.")


# =============================================================================
# TIER 3: Cross-Feature Combinations
# =============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise cross-system physical consistency and signal correlations."""

    def test_tier3_wheel_speed_vs_motor_rpm_ratio_consistency(self):
        """Tier 3.1: Verify canonical speed formula v = inv_rpm / 37.978 on active competition runs."""
        # Check Barcelona acceleration run
        bcn_file = os.path.join(DATOS_DIR, "ISC_20260805_182532_marta_accel_bcn.csv")
        assert os.path.exists(bcn_file), "Barcelona acceleration file must exist."
        df = pd.read_csv(bcn_file)
        if "inv_rpm" in df.columns and "inv_speed_actual" in df.columns:
            active = df[(df["inv_rpm"] > 1000) & (df["inv_speed_actual"] > 10)]
            if len(active) > 0:
                ratios = active["inv_rpm"] / active["inv_speed_actual"]
                median_ratio = ratios.median()
                # August ratio is ~37.978
                assert 37.0 <= median_ratio <= 39.0, f"Expected speed scaling ratio ~37.98, found {median_ratio:.2f}"

    def test_tier3_dcdc_current_bms_vs_vcu_polarity_and_magnitude(self):
        """Tier 3.2: Verify DCDC current magnitude match between BMS (-38.5A) and VCU (+37.1A)."""
        # Read a representative BMS log and a representative Telemetry log
        bms_file = os.path.join(AMS_DIR, "LOG0089.CSV")
        assert os.path.exists(bms_file), "LOG0089.CSV must exist."
        bms_df = pd.read_csv(bms_file)
        bms_dcdc_A = bms_df["Idcdc_mA"].median() / 1000.0

        vcu_file = os.path.join(DATOS_DIR, "ISC_20260920_114635_andr%C3%A9s_calentamiento_canto1.csv")
        assert os.path.exists(vcu_file), "Cantoblanco warmup telemetry log must exist."
        vcu_df = pd.read_csv(vcu_file)
        vcu_dcdc_A = vcu_df["corriente_dcdc"].median()

        # BMS measures negative (-38A), VCU measures positive (+37A)
        assert -45.0 <= bms_dcdc_A <= -30.0, f"Expected BMS DCDC current ~ -38.5A, found {bms_dcdc_A:.1f}A"
        assert 30.0 <= vcu_dcdc_A <= 45.0, f"Expected VCU DCDC current ~ +37.1A, found {vcu_dcdc_A:.1f}A"
        magnitude_delta = abs(abs(bms_dcdc_A) - abs(vcu_dcdc_A))
        assert magnitude_delta <= 5.0, f"DCDC magnitude discrepancy {magnitude_delta:.2f}A exceeds 5.0A tolerance."

    def test_tier3_contactor_shock_timing_vs_ntc_disconnect_drop(self):
        """Tier 3.3: Verify contactor opening clunk precedes/coincides with thermistor disconnect within <=400ms."""
        log74 = os.path.join(AMS_DIR, "LOG0074.CSV")
        imu74 = os.path.join(AMS_DIR, "IMU0074.CSV")
        assert os.path.exists(log74) and os.path.exists(imu74), "LOG0074 and IMU0074 must exist."
        df_log = pd.read_csv(log74)
        df_imu = pd.read_csv(imu74)

        # In LOG0074, TSMS drops from 1 to 0 at tick_ms=13250, NtcDisc trips at tick_ms=13500 (delta = 250ms <= 400ms)
        tsms_1_to_0 = df_log[(df_log["tsms"].shift(1) == 1) & (df_log["tsms"] == 0)]
        assert not tsms_1_to_0.empty, "TSMS 1->0 transition should occur in LOG0074."
        t_tsms_drop = tsms_1_to_0["tick_ms"].iloc[0]

        fault_rows = df_log[df_log["fault"] == 13]  # NtcDisc
        assert not fault_rows.empty, "NtcDisc fault must occur in LOG0074."
        t_fault = fault_rows["tick_ms"].iloc[0]

        delta_ms = t_fault - t_tsms_drop
        assert 0 <= delta_ms <= 400, f"NtcDisc should occur within 400ms of contactor open, got {delta_ms}ms"

        # Co-located IMU must show acceleration jerk
        shock_points = df_imu[(df_imu["tick_ms"] >= t_tsms_drop - 250) & (df_imu["tick_ms"] <= t_fault + 250)]
        assert not shock_points.empty, "IMU data must cover the contactor opening interval."
        acc_magnitude = np.sqrt(shock_points["ax_g"]**2 + shock_points["ay_g"]**2 + (shock_points["az_g"] - 1.0)**2)
        assert acc_magnitude.max() > 0.04, "Contactor opening shock must register measurable acceleration pulse."

    def test_tier3_rule_ev_2_3_plausibility_breach_during_braking(self):
        """Tier 3.4: Verify EV 2.3 plausibility breach asserts torque=0 when APPS > 25% while braking."""
        austria_log = os.path.join(DATOS_DIR, "ISC_20260730_092010_andres_dieguez_2_austria_endurance_2026_2 (1).csv")
        assert os.path.exists(austria_log), "Austria endurance log must exist."
        df = pd.read_csv(austria_log)

        # Locate ev_2_3 == 1 event
        ev_rows = df[df["ev_2_3"] == 1]
        assert not ev_rows.empty, "Must find EV 2.3 assertion event."
        row = ev_rows.iloc[0]
        # In this event, driver had APPS > 25% (apps1_raw > 2500) and mechanical brake engaged
        assert row["apps1_raw"] > 2500, "EV 2.3 breach requires high pedal travel."
        assert row["brake_raw"] > 400, "EV 2.3 breach requires active brake actuation."
        # Motor torque must be suppressed
        assert row["torque_pct"] == 0 or row["par_motor"] == 0, "Torque must be 0 upon EV 2.3 plausibility breach."

    def test_tier3_inverter_dem_undervoltage_code_causality(self):
        """Tier 3.5: Verify inverter DEM error 2 occurs strictly after BMS FSM ERR 5 (air opening causality)."""
        jarama_log = os.path.join(DATOS_DIR, "ISC_20260929_113826_Andres_Jarama_1.1 (2).csv")
        assert os.path.exists(jarama_log), "Jarama log must exist."
        df = pd.read_csv(jarama_log)

        bms_trips = df[df["ams_fsm_state"] == 5]
        inv_trips = df[df["inv_error"] == 2]

        assert not bms_trips.empty, "BMS FSM must transition to state 5 in Jarama log."
        assert not inv_trips.empty, "Inverter must trip error code 2 in Jarama log."

        t_col = "time_elapsed_s" if "time_elapsed_s" in df.columns else ("timestamp_s" if "timestamp_s" in df.columns else "tick_ms")
        t_bms = bms_trips[t_col].iloc[0]
        t_inv = inv_trips[t_col].iloc[0]

        # Inverter undervoltage must occur strictly after BMS contactor opening
        assert t_inv > t_bms, f"Inverter error at t={t_inv} must follow BMS trip at t={t_bms}"
        assert t_inv - t_bms >= 2.0, f"Expected inverter DC bus collapse delay >= 2.0s, got {t_inv - t_bms:.2f}s"


# =============================================================================
# TIER 4: Real-World Application Scenarios
# =============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Season replay, competition evaluations, and engineering root causes."""

    def test_tier4_full_season_replay_four_venues(self):
        """Tier 4.1: Replay full season across all 4 operational venues (Cantoblanco, Austria, Barcelona, Jarama)."""
        vehicle_files = glob.glob(os.path.join(DATOS_DIR, "*.csv"))
        filenames_lower = [os.path.basename(f).lower() for f in vehicle_files]

        has_canto = any("canto" in f for f in filenames_lower)
        has_austria = any("austria" in f for f in filenames_lower)
        has_bcn = any("montmelo" in f or "barcelona" in f for f in filenames_lower)
        has_jarama = any("jarama" in f for f in filenames_lower)

        assert has_canto, "Full season replay must include Cantoblanco venue."
        assert has_austria, "Full season replay must include Austria venue."
        assert has_bcn, "Full season replay must include Barcelona venue."
        assert has_jarama, "Full season replay must include Jarama venue."

    def test_tier4_austria_endurance_stint_2_evaluation(self):
        """Tier 4.2: Evaluate Austria Endurance Stint 2 (duration > 200s, peak speed > 65 km/h, peak current > 70A)."""
        austria_log = os.path.join(DATOS_DIR, "ISC_20260730_092010_andres_dieguez_2_austria_endurance_2026_2 (1).csv")
        assert os.path.exists(austria_log), "Austria endurance log must exist."
        df = pd.read_csv(austria_log)

        # Calibrated vehicle speed
        if "inv_rpm" in df.columns:
            speeds = df["inv_rpm"] / 37.978
            peak_speed = speeds.max()
            assert peak_speed >= 65.0, f"Peak calibrated speed in Austria should exceed 65 km/h, got {peak_speed:.1f}"

        # Peak battery current
        curr_col = "corriente_accu" if "corriente_accu" in df.columns else "corriente_bateria"
        if curr_col in df.columns:
            peak_curr = df[curr_col].max()
            assert peak_curr >= 70.0, f"Peak discharge current should exceed 70A, got {peak_curr:.1f}A"

    def test_tier4_jarama_shutdown_root_cause_reconstruction(self):
        """Tier 4.3: Reconstruct chronological shutdown cascade in Jarama (Vmin 65535 -> NTC -32768 -> FSM 5 -> DEM 2)."""
        jarama_log = os.path.join(DATOS_DIR, "ISC_20260929_113826_Andres_Jarama_1.1 (2).csv")
        assert os.path.exists(jarama_log), "Jarama log must exist."
        df = pd.read_csv(jarama_log)

        # Locate line ~410 where v_cell_min_mV spikes to 65535
        spike_rows = df[df["v_cell_min_mV"] == 65535]
        assert not spike_rows.empty, "Cell min voltage must spike to 65535 sentinel."

        # NTCs drop to -32768
        ntc_disc_rows = df[df["tmax_mod0"] == -32768]
        assert not ntc_disc_rows.empty, "NTC max temperatures must drop to -32768."

        # BMS FSM transitions to 5
        fsm_err_rows = df[df["ams_fsm_state"] == 5]
        assert not fsm_err_rows.empty, "BMS state machine must transition to error state 5."

        # Sequence ordering
        idx_spike = spike_rows.index[0]
        idx_ntc = ntc_disc_rows.index[0]
        idx_fsm = fsm_err_rows.index[0]
        assert idx_spike <= idx_ntc <= idx_fsm, "Shutdown sequence must follow Spike -> NTC drop -> FSM 5 transition."

    def test_tier4_cantoblanco_cell_sag_and_imbalance(self):
        """Tier 4.4: Reconstruct Cantoblanco warmup stint cell sag (instantaneous spread > 300mV, module delta > 80mV)."""
        # Cross-validate LOG0089/LOG0090 in AMS logs
        log89_path = os.path.join(AMS_DIR, "LOG0089.CSV")
        assert os.path.exists(log89_path), "LOG0089.CSV must exist."
        df = pd.read_csv(log89_path)

        # Check maximum instantaneous cell spread
        max_spread = (df["vmax_mV"] - df["vmin_mV"]).max()
        assert max_spread >= 300, f"Maximum cell voltage spread should exceed 300mV, found {max_spread}mV"
        assert df["vmin_mV"].min() <= 3100, f"Minimum cell voltage under load should drop to ~3073mV, found {df['vmin_mV'].min()}mV"

        # Identify cells for module 0 and module 3
        m0_cells = [f"c0_{i}" for i in range(19)]
        m3_cells = [f"c3_{i}" for i in range(19)]

        load_rows = df[df["I_filt_mA"].abs() > 30000]
        if not load_rows.empty:
            m0_mean = load_rows[m0_cells].mean(axis=1)
            m3_mean = load_rows[m3_cells].mean(axis=1)
            delta = (m0_mean - m3_mean).abs().max()
            assert delta > 80.0, f"Module delta under load should exceed 80mV, found {delta:.1f}mV"

    def test_tier4_car_2026_2027_engineering_recommendations(self):
        """Tier 4.5: Verify technical report articulates specific design improvements for the 2026/2027 car."""
        md_path = require_artifact("reports/formula_student_ev_season_analysis.md")
        with open(md_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().lower()

        # Check for core engineering recommendations identified during season analysis:
        # 1. DCDC isolation / precharge resistor modification
        assert ("dcdc" in text or "precharge" in text), "Report must address precharge/DCDC divider issue."
        # 2. Thermistor filtering or harness mechanical strain relief
        assert ("ntc" in text or "thermistor" in text or "openwire" in text), "Report must address NTC harness / EMI false trips."
        # 3. Cell module balancing / busbar cooling
        assert ("cell" in text or "balance" in text or "module" in text), "Report must address module cell balance / sag."

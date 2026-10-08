"""Adversarial Empirical Challenger Test Suite for Milestone 2.

Author: M2 Challenger 2 (Empirical Plausibility & Taxonomy Verifier)
Target: artifacts/fault_classification_table.csv, artifacts/consolidated_fault_log.csv,
        Rule EV 2.3 & Rule T11.8.9 compliance algorithms.

Verifies:
1. Complete taxonomy table (4 categories, 12 modes, complete schemas, actionable mitigations)
2. Consolidated fault log (18 records, exact timestamps, pre-fault condition JSONs, physical root causes)
3. Rule EV 2.3 boundary conditions, latching/unlatching dynamics, and synthetic stress harnesses
4. Rule T11.8.9 discrepancy thresholds, 100ms persistence watchdog, rate jitter, and resets
5. Empirical validation against physical season telemetry (Austria EV 2.3, season-wide T11.8.9 scan)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.rules import (
    APPS_DISCREPANCY_MAX_PCT,
    APPS_DISCREPANCY_PERSIST_S,
    APPS_RESET_THRESHOLD_PCT,
    APPS_TRIP_THRESHOLD_PCT,
    BRAKE_TRIP_RAW_THRESHOLD,
    RuleBreachEvent,
    RulePlausibilityValidator,
    evaluate_rule_ev_2_3,
    evaluate_rule_t11_8_9,
    export_fault_classification_table,
)
from src.diagnostics.fault_analyzer import (
    PrechargeAnalyzer,
    SafetyFaultAnalyzer,
)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
AMS_DIR = BASE_DIR / "AMS logs" / "AMS logs"
DATOS_DIR = BASE_DIR / "datos2526"


# ===========================================================================
# 1. TAXONOMY COMPLETENESS & INVARIANTS (fault_classification_table.csv)
# ===========================================================================

class TestTaxonomyCompletenessEmpirical:
    """Rigorous empirical validation of the 4-tier, 12-mode taxonomy table."""

    @pytest.fixture(scope="class")
    def taxonomy_df(self) -> pd.DataFrame:
        csv_path = ARTIFACTS_DIR / "fault_classification_table.csv"
        assert csv_path.exists(), f"Missing required artifact: {csv_path}"
        df = pd.read_csv(csv_path)
        return df

    def test_taxonomy_exact_cardinality(self, taxonomy_df: pd.DataFrame):
        """Must contain exactly 12 fault modes across exactly 4 categories."""
        assert len(taxonomy_df) == 12, f"Expected 12 fault modes, got {len(taxonomy_df)}"
        
        expected_categories = {
            "Hardware Fault",
            "Operational / Algorithmic Cutoff",
            "Sensor / EMI Artifact",
            "Driver / Regulatory Plausibility",
        }
        actual_categories = set(taxonomy_df["category"].unique())
        assert actual_categories == expected_categories, f"Category mismatch: {actual_categories}"

        counts = taxonomy_df["category"].value_counts().to_dict()
        assert counts["Hardware Fault"] == 4, "Must have exactly 4 Hardware Faults"
        assert counts["Operational / Algorithmic Cutoff"] == 3, "Must have exactly 3 Operational Cutoffs"
        assert counts["Sensor / EMI Artifact"] == 3, "Must have exactly 3 Sensor/EMI Artifacts"
        assert counts["Driver / Regulatory Plausibility"] == 2, "Must have exactly 2 Regulatory Plausibility modes"

    def test_taxonomy_mode_ids_and_prefixes(self, taxonomy_df: pd.DataFrame):
        """Must have standard FM_* identifiers matching their category prefixes."""
        expected_ids = [
            "FM_HW_01", "FM_HW_02", "FM_HW_03", "FM_HW_04",
            "FM_OP_01", "FM_OP_02", "FM_OP_03",
            "FM_EMI_01", "FM_EMI_02", "FM_EMI_03",
            "FM_REG_01", "FM_REG_02",
        ]
        assert list(taxonomy_df["fault_mode_id"]) == expected_ids
        assert taxonomy_df["fault_mode_id"].is_unique

    def test_taxonomy_schema_and_zero_nulls(self, taxonomy_df: pd.DataFrame):
        """Schema must contain 9 mandatory columns with zero null or empty values."""
        mandatory_cols = [
            "fault_mode_id", "category", "fault_name", "subsystem",
            "primary_vs_secondary", "affected_sessions", "detection_criteria",
            "root_cause_mechanism", "recommended_mitigation",
        ]
        assert list(taxonomy_df.columns) == mandatory_cols

        # Check for NaNs or whitespace-only cells
        for col in mandatory_cols:
            assert taxonomy_df[col].isna().sum() == 0, f"Column {col} contains NaN values"
            empty_strings = taxonomy_df[col].astype(str).str.strip().eq("").sum()
            assert empty_strings == 0, f"Column {col} contains empty/whitespace strings"

    def test_taxonomy_actionable_engineering_mitigations(self, taxonomy_df: pd.DataFrame):
        """Mitigations must provide specific, actionable engineering recommendations."""
        keywords = {
            "FM_HW_01": ["connector", "debounce"],
            "FM_HW_02": ["mask", "t0_12", "pcb"],
            "FM_HW_03": ["isospi", "shielded", "locking"],
            "FM_HW_04": ["secondary", "inverter", "bms"],
            "FM_OP_01": ["dcdc", "relay", "precharge"],
            "FM_OP_02": ["vcu", "limiter", "margin"],
            "FM_OP_03": ["balance", "pouch", "derating"],
            "FM_EMI_01": ["flyback", "persistence", "bounce"],
            "FM_EMI_02": ["shock", "isolate", "damper"],
            "FM_EMI_03": ["filter", "ltc", "diagnostic"],
            "FM_REG_01": ["pedal", "chime", "training"],
            "FM_REG_02": ["calibration", "apps", "scrutineering"],
        }
        for _, row in taxonomy_df.iterrows():
            f_id = row["fault_mode_id"]
            mitigation_lower = row["recommended_mitigation"].lower()
            expected_kw = keywords[f_id]
            matched = any(kw in mitigation_lower for kw in expected_kw)
            assert matched, f"Mitigation for {f_id} lacks technical specificity: {row['recommended_mitigation']}"


# ===========================================================================
# 2. CONSOLIDATED FAULT LOG VERIFICATION (consolidated_fault_log.csv)
# ===========================================================================

class TestConsolidatedFaultLogEmpirical:
    """Rigorous verification of exact timestamps, root causes, and pre-fault conditions."""

    @pytest.fixture(scope="class")
    def fault_df(self) -> pd.DataFrame:
        csv_path = ARTIFACTS_DIR / "consolidated_fault_log.csv"
        assert csv_path.exists(), f"Missing required artifact: {csv_path}"
        df = pd.read_csv(csv_path)
        return df

    def test_fault_log_record_count_and_columns(self, fault_df: pd.DataFrame):
        """Must contain 18 records (13 BMS + 5 telemetry) with complete schema."""
        assert len(fault_df) == 18, f"Expected 18 fault records, got {len(fault_df)}"
        mandatory_cols = [
            "timestamp", "timestamp_s", "session_id", "log_file", "subsystem",
            "fault_name", "fault_code", "fault_type", "primary_flag",
            "pre_fault_metrics", "root_cause",
        ]
        assert list(fault_df.columns) == mandatory_cols
        assert fault_df.isna().sum().sum() == 0, "Consolidated fault log contains null values"

    def test_fault_log_primary_vs_secondary_balance(self, fault_df: pd.DataFrame):
        """Exactly 15 primary root cause events and 3 secondary/continuation events."""
        primary_counts = fault_df["primary_flag"].value_counts().to_dict()
        assert primary_counts[True] == 15, "Expected 15 primary root causes"
        assert primary_counts[False] == 3, "Expected 3 secondary/continuation events"

        # The three secondary events must be LOG0103 (rollover) and two Inverter DEM 2 events
        secondary_records = fault_df[fault_df["primary_flag"] == False]
        assert "LOG0103.CSV" in secondary_records["log_file"].values
        dem2_records = secondary_records[secondary_records["fault_name"] == "Inverter_UV_DEM2"]
        assert len(dem2_records) == 2

    def test_verify_13_bms_shutdowns_against_raw_files(self, fault_df: pd.DataFrame):
        """Cross-correlates all 13 BMS shutdown records directly against raw CSVs in AMS logs."""
        bms_records = fault_df.head(13)

        for _, row in bms_records.iterrows():
            raw_path = AMS_DIR / row["log_file"]
            assert raw_path.exists(), f"Raw BMS log {raw_path} not found on disk"
            df_raw = pd.read_csv(raw_path)

            # Find first row with fsm == 5
            err_rows = df_raw[df_raw["fsm"] == 5]
            assert not err_rows.empty, f"No fsm=5 error state in {row['log_file']}"
            first_err = err_rows.iloc[0]

            expected_time_s = round(float(first_err["tick_ms"] / 1000.0), 2)
            assert np.isclose(row["timestamp_s"], expected_time_s, atol=0.01), (
                f"Timestamp mismatch in {row['log_file']}: logged {row['timestamp_s']}, raw {expected_time_s}"
            )
            assert row["fault_code"] == int(first_err["fault"]), (
                f"Fault code mismatch in {row['log_file']}: logged {row['fault_code']}, raw {first_err['fault']}"
            )

    def test_pre_fault_metrics_json_integrity(self, fault_df: pd.DataFrame):
        """Validates that all pre_fault_metrics are valid JSON with physical numerical bounds."""
        for idx, row in fault_df.iterrows():
            metrics_str = row["pre_fault_metrics"]
            data = json.loads(metrics_str)
            assert isinstance(data, dict), f"Row {idx} metrics is not a valid JSON dictionary"

            if "pack_V" in data:
                assert 250.0 <= data["pack_V"] <= 400.0, f"Unphysical pack voltage in row {idx}: {data['pack_V']}"
            if "dcbus_V" in data:
                assert 0.0 <= data["dcbus_V"] <= 400.0, f"Unphysical DC bus voltage in row {idx}: {data['dcbus_V']}"
            if "I_filt_A" in data:
                assert -50.0 <= data["I_filt_A"] <= 250.0, f"Unphysical filtered current in row {idx}: {data['I_filt_A']}"
            if "cell_spread_mV" in data:
                assert 0.0 <= data["cell_spread_mV"] <= 1000.0, f"Unphysical cell spread in row {idx}: {data['cell_spread_mV']}"

    def test_verify_telemetry_fault_records(self, fault_df: pd.DataFrame):
        """Cross-correlates the 5 telemetry fault records against raw CSV files in datos2526."""
        telemetry_records = fault_df.tail(5)

        # 1. Jarama Primary Comms Fault & Secondary Inverter DEM2
        jarama_p = DATOS_DIR / "ISC_20260929_113826_Andres_Jarama_1.1 (2).csv"
        assert jarama_p.exists()
        df_j = pd.read_csv(jarama_p)
        j_primary = telemetry_records[telemetry_records["fault_name"] == "BmsCommsFault"].iloc[0]
        j_secondary = telemetry_records[(telemetry_records["fault_name"] == "Inverter_UV_DEM2") & (telemetry_records["session_id"].str.contains("Jarama"))].iloc[0]
        
        # Inverter DEM 2 must occur ~5s after BMS shutdown
        delta_j = j_secondary["timestamp_s"] - j_primary["timestamp_s"]
        assert 4.5 <= delta_j <= 5.5, f"Expected ~5s delay in Jarama DEM2, got {delta_j}"

        # 2. Cantoblanco Primary Trip & Secondary Inverter DEM2
        canto_p = DATOS_DIR / "ISC_20260920_154755_andres_endurance_p1_canto.csv"
        assert canto_p.exists()
        c_primary = telemetry_records[telemetry_records["fault_name"] == "BmsShutdown"].iloc[0]
        c_secondary = telemetry_records[(telemetry_records["fault_name"] == "Inverter_UV_DEM2") & (telemetry_records["session_id"].str.contains("canto"))].iloc[0]
        delta_c = c_secondary["timestamp_s"] - c_primary["timestamp_s"]
        assert 2.5 <= delta_c <= 3.5, f"Expected ~3.2s delay in Cantoblanco DEM2, got {delta_c}"

        # 3. Austria Endurance Stint 2 Regulatory Cutoff
        austria_p = DATOS_DIR / "ISC_20260730_092010_andres_dieguez_2_austria_endurance_2026_2 (1).csv"
        assert austria_p.exists()
        df_a = pd.read_csv(austria_p)
        a_record = telemetry_records[telemetry_records["fault_name"] == "EV_2_3_APPS_Brake"].iloc[0]
        assert np.isclose(a_record["timestamp_s"], 220.50, atol=0.1)


# ===========================================================================
# 3. RULE EV 2.3 BOUNDARY & CORNER CASE STRESS-TESTING
# ===========================================================================

class TestRuleEV23BoundaryAndCornerCases:
    """Stress-tests Rule EV 2.3 APPS/Brake Plausibility algorithm under adversarial edge cases."""

    def test_ev_2_3_exact_threshold_boundary(self):
        """Verifies boundary trip conditions: APPS strictly > 25.0% and Brake > 600 raw."""
        # Case A: APPS exactly 25.0% with brake pressed -> NO TRIP
        df_border_sub = pd.DataFrame({
            "timestamp_s": [0.0, 0.2, 0.4],
            "apps1_pct": [25.0, 25.0, 25.0],
            "apps2_pct": [25.0, 25.0, 25.0],
            "brake_pct": [10.0, 10.0, 10.0],
            "brake_raw": [700.0, 700.0, 700.0],
            "torque_pct": [20.0, 20.0, 20.0],
        })
        latch, events = evaluate_rule_ev_2_3(df_border_sub)
        assert latch.sum() == 0, "APPS exactly 25.0% must NOT trigger EV 2.3 breach"
        assert len(events) == 0

        # Case B: APPS 25.01% with brake pressed -> TRIPS
        df_border_sup = pd.DataFrame({
            "timestamp_s": [0.0, 0.2, 0.4],
            "apps1_pct": [25.01, 25.01, 25.01],
            "apps2_pct": [25.01, 25.01, 25.01],
            "brake_pct": [10.0, 10.0, 10.0],
            "brake_raw": [700.0, 700.0, 700.0],
            "torque_pct": [0.0, 0.0, 0.0],
        })
        latch, events = evaluate_rule_ev_2_3(df_border_sup)
        assert (latch == 1).all(), "APPS 25.01% with brake must trigger EV 2.3 breach"
        assert len(events) >= 1

        # Case C: APPS 50% with brake exactly 600.0 raw and brake_pct 2.0% -> NO TRIP
        df_brake_border = pd.DataFrame({
            "timestamp_s": [0.0, 0.2, 0.4],
            "apps1_pct": [50.0, 50.0, 50.0],
            "apps2_pct": [50.0, 50.0, 50.0],
            "brake_pct": [2.0, 2.0, 2.0],
            "brake_raw": [600.0, 600.0, 600.0],
            "torque_pct": [50.0, 50.0, 50.0],
        })
        latch, events = evaluate_rule_ev_2_3(df_brake_border)
        assert latch.sum() == 0, "Brake raw 600.0 and pct 2.0 must NOT trigger EV 2.3 breach"

        # Case D: APPS 50% with brake 600.1 raw -> TRIPS
        df_brake_trip = pd.DataFrame({
            "timestamp_s": [0.0, 0.2, 0.4],
            "apps1_pct": [50.0, 50.0, 50.0],
            "apps2_pct": [50.0, 50.0, 50.0],
            "brake_pct": [2.0, 2.0, 2.0],
            "brake_raw": [600.1, 600.1, 600.1],
            "torque_pct": [0.0, 0.0, 0.0],
        })
        latch, events = evaluate_rule_ev_2_3(df_brake_trip)
        assert (latch == 1).all(), "Brake raw 600.1 with APPS 50% must trigger EV 2.3 breach"

    def test_ev_2_3_latching_and_unlatching_hysteresis(self):
        """Verifies latching persistence between 5% and 25%, clearing strictly at < 5%."""
        times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        # Profile:
        # t=0.0: normal driving (APPS 20%, Brake 0) -> Clean
        # t=0.2: trip event (APPS 50%, Brake 700) -> Trips, latches
        # t=0.4: driver releases brake but keeps APPS 40% -> Remains latched!
        # t=0.6: driver drops APPS to 20% (brake 0) -> Remains latched!
        # t=0.8: driver drops APPS to 10% (brake 0) -> Remains latched!
        # t=1.0: driver drops APPS to 5.0% (brake 0) -> Boundary: remains latched (< 5.0 required!)
        # t=1.2: driver drops APPS to 4.9% (brake 0) -> Latch CLEARS!
        # t=1.4: driver resumes throttle to 20% (brake 0) -> Remains clean!
        # t=1.6: driver presses throttle to 60% (brake 0) -> Remains clean!
        df = pd.DataFrame({
            "timestamp_s": times,
            "time_elapsed_s": times,
            "apps1_pct": [20.0, 50.0, 40.0, 20.0, 10.0, 5.0, 4.9, 20.0, 60.0],
            "apps2_pct": [20.0, 50.0, 40.0, 20.0, 10.0, 5.0, 4.9, 20.0, 60.0],
            "brake_pct": [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "brake_raw": [400.0, 700.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0],
            "torque_pct": [20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 60.0],
        })

        latch, events = evaluate_rule_ev_2_3(df)
        expected_latch = [0, 1, 1, 1, 1, 1, 0, 0, 0]
        assert list(latch) == expected_latch, f"Latching series mismatch: got {list(latch)}"
        assert len(events) == 1
        ev = events[0]
        assert ev.time_elapsed_s == 0.2
        assert ev.latch_cleared_s == 1.2
        assert np.isclose(ev.latch_duration_s, 1.0, atol=0.01)

    def test_ev_2_3_multi_cycle_recurrent_breaches(self):
        """Verifies multiple sequential breach-clear-breach cycles in a single session."""
        times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        # Cycle 1: trip at 0.2, clear at 0.4
        # Clean at 0.6
        # Cycle 2: trip at 0.8, clear at 1.2
        # Clean at 1.4
        df = pd.DataFrame({
            "timestamp_s": times,
            "time_elapsed_s": times,
            "apps1_pct": [10.0, 60.0, 2.0, 15.0, 70.0, 30.0, 1.0, 20.0],
            "apps2_pct": [10.0, 60.0, 2.0, 15.0, 70.0, 30.0, 1.0, 20.0],
            "brake_pct": [0.0, 15.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0],
            "brake_raw": [400.0, 800.0, 400.0, 400.0, 800.0, 400.0, 400.0, 400.0],
            "torque_pct": [10.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 20.0],
        })

        latch, events = evaluate_rule_ev_2_3(df)
        assert list(latch) == [0, 1, 0, 0, 1, 1, 0, 0]
        assert len(events) == 2
        assert events[0].time_elapsed_s == 0.2
        assert events[0].latch_cleared_s == 0.4
        assert events[1].time_elapsed_s == 0.8
        assert events[1].latch_cleared_s == 1.2

    def test_ev_2_3_unresolved_latch_at_end_of_log(self):
        """If log terminates while latched, event is still recorded with latch_cleared_s=None."""
        df = pd.DataFrame({
            "timestamp_s": [0.0, 0.2, 0.4],
            "time_elapsed_s": [0.0, 0.2, 0.4],
            "apps1_pct": [10.0, 60.0, 50.0],
            "apps2_pct": [10.0, 60.0, 50.0],
            "brake_pct": [0.0, 10.0, 0.0],
            "brake_raw": [400.0, 700.0, 400.0],
            "torque_pct": [10.0, 0.0, 0.0],
        })
        latch, events = evaluate_rule_ev_2_3(df)
        assert list(latch) == [0, 1, 1]
        assert len(events) == 1
        assert events[0].latch_cleared_s is None
        assert events[0].latch_duration_s is None

    def test_ev_2_3_robustness_empty_and_single_row(self):
        """Handles empty and 1-row DataFrames gracefully without exceptions."""
        df_empty = pd.DataFrame()
        latch_empty, events_empty = evaluate_rule_ev_2_3(df_empty)
        assert len(latch_empty) == 0
        assert len(events_empty) == 0

        df_single = pd.DataFrame({
            "timestamp_s": [0.0],
            "apps1_pct": [10.0],
            "apps2_pct": [10.0],
            "brake_pct": [0.0],
            "brake_raw": [400.0],
            "torque_pct": [10.0],
        })
        latch_single, events_single = evaluate_rule_ev_2_3(df_single)
        assert len(latch_single) == 1
        assert latch_single.iloc[0] == 0
        assert len(events_single) == 0


# ===========================================================================
# 4. RULE T11.8.9 BOUNDARY & CORNER CASE STRESS-TESTING
# ===========================================================================

class TestRuleT1189BoundaryAndCornerCases:
    """Stress-tests Rule T11.8.9 APPS Dual-Sensor Rationality algorithm."""

    def test_t11_8_9_exact_discrepancy_boundary(self):
        """Discrepancy exactly 10.0% must NOT trip even if persistent; 10.01% trips."""
        # 200 ms timestep (duration between samples is 0.2s > 0.1s persistence watchdog)
        # Case A: exactly 10.00% difference
        df_border_sub = pd.DataFrame({
            "timestamp_s": [0.0, 0.2, 0.4],
            "apps1_pct": [50.0, 50.0, 50.0],
            "apps2_pct": [40.0, 40.0, 40.0],  # delta = 10.00%
        })
        trips, events = evaluate_rule_t11_8_9(df_border_sub)
        assert trips.sum() == 0, "|APPS1 - APPS2| exactly 10.0% must NOT trip"
        assert len(events) == 0

        # Case B: 10.01% difference
        df_border_sup = pd.DataFrame({
            "timestamp_s": [0.0, 0.2, 0.4],
            "apps1_pct": [50.0, 50.0, 50.0],
            "apps2_pct": [39.99, 39.99, 39.99],  # delta = 10.01%
        })
        trips, events = evaluate_rule_t11_8_9(df_border_sup)
        # sample 0 starts timer at 0.0; sample 1 at 0.2 has duration 0.2 > 0.1 -> trips!
        assert list(trips) == [0, 1, 1], "Delta 10.01% exceeding 100ms must trip"
        assert len(events) == 2

    def test_t11_8_9_persistence_watchdog_timing_high_rate(self):
        """Simulates 100 Hz (10 ms) telemetry: transient spike < 100 ms ignored, > 100 ms trips."""
        # 15 time steps @ 10 ms (0.00 to 0.14s)
        # Delta > 10% active from sample 1 (0.01s) to sample 10 (0.10s): duration = 0.09s <= 0.10s -> NO TRIP
        # Then delta drops back to 0 at sample 11 (0.11s)
        times_short = np.arange(15) * 0.01
        apps1_short = np.full(15, 50.0)
        apps2_short = np.full(15, 50.0)
        apps2_short[1:10] = 30.0  # 20% delta for 90 ms

        df_short = pd.DataFrame({
            "timestamp_s": times_short,
            "apps1_pct": apps1_short,
            "apps2_pct": apps2_short,
        })
        trips_short, events_short = evaluate_rule_t11_8_9(df_short)
        assert trips_short.sum() == 0, "Transient discrepancy <= 100 ms must NOT trip"
        assert len(events_short) == 0

        # Now test duration = 120 ms (samples 1 to 13):
        # sample 1: start_t = 0.01
        # sample 11: t = 0.11, duration = 0.10 (not > 0.10)
        # sample 12: t = 0.12, duration = 0.11 > 0.10 -> TRIPS!
        # sample 13: t = 0.13, duration = 0.12 > 0.10 -> TRIPS!
        times_long = np.arange(16) * 0.01
        apps1_long = np.full(16, 50.0)
        apps2_long = np.full(16, 50.0)
        apps2_long[1:14] = 30.0

        df_long = pd.DataFrame({
            "timestamp_s": times_long,
            "apps1_pct": apps1_long,
            "apps2_pct": apps2_long,
        })
        trips_long, events_long = evaluate_rule_t11_8_9(df_long)
        assert trips_long.iloc[12] == 1, "Discrepancy persisting > 100ms must trip"
        assert trips_long.iloc[13] == 1
        assert len(events_long) == 2

    def test_t11_8_9_watchdog_timer_resets_after_transient(self):
        """Watchdog timer must completely reset when discrepancy drops below 10%, not accumulate."""
        # Two separate 60 ms spikes separated by 100 ms of normal signal:
        # Total time > 100 ms, but each transient is <= 100 ms -> NO TRIP
        times = np.arange(25) * 0.01
        apps1 = np.full(25, 50.0)
        apps2 = np.full(25, 50.0)
        # Spike 1: samples 2 to 7 (60 ms)
        apps2[2:8] = 35.0
        # Normal: samples 8 to 15 (80 ms)
        # Spike 2: samples 16 to 21 (60 ms)
        apps2[16:22] = 35.0

        df = pd.DataFrame({
            "timestamp_s": times,
            "apps1_pct": apps1,
            "apps2_pct": apps2,
        })
        trips, events = evaluate_rule_t11_8_9(df)
        assert trips.sum() == 0, "Non-contiguous spikes must NOT accumulate watchdog time"
        assert len(events) == 0

    def test_t11_8_9_channel_inversion_symmetry(self):
        """Verifies mathematical symmetry: APPS1 > APPS2 behaves identically to APPS2 > APPS1."""
        df_a = pd.DataFrame({
            "timestamp_s": [0.0, 0.2, 0.4],
            "apps1_pct": [60.0, 60.0, 60.0],
            "apps2_pct": [40.0, 40.0, 40.0],
        })
        df_b = pd.DataFrame({
            "timestamp_s": [0.0, 0.2, 0.4],
            "apps1_pct": [40.0, 40.0, 40.0],
            "apps2_pct": [60.0, 60.0, 60.0],
        })
        trips_a, events_a = evaluate_rule_t11_8_9(df_a)
        trips_b, events_b = evaluate_rule_t11_8_9(df_b)
        assert list(trips_a) == list(trips_b)
        assert len(events_a) == len(events_b)


# ===========================================================================
# 5. PHYSICAL SEASON TELEMETRY REPLAY & EMPIRICAL PLACEMENT
# ===========================================================================

class TestPhysicalSeasonReplayEmpirical:
    """Verifies physical session datasets: Austria Endurance EV 2.3 and season T11.8.9 scan."""

    def test_austria_endurance_stint_2_breach_replay(self):
        """Empirically replays Austria Endurance Stint 2 and verifies Row 149 breach & latch."""
        validator = RulePlausibilityValidator(DATOS_DIR)
        res = validator.validate_austria_endurance_stint_2()

        assert res["total_samples"] == 170
        assert res["breach_events_count"] == 1
        assert res["ev_2_3_flag_count"] >= 1

        ev = res["breach_details"][0]
        # Verified exact values at row 149
        assert ev["rule_name"] == "EV 2.3 (APPS/Brake Plausibility)"
        assert np.isclose(ev["time_elapsed_s"], 220.504, atol=0.1)
        assert ev["apps1_raw"] == 3382.0
        assert ev["brake_raw"] == 607.0
        assert ev["torque_pct"] == 0.0
        assert ev["is_torque_suppressed"] is True
        assert ev["latch_cleared_s"] is not None
        assert ev["latch_duration_s"] > 5.0, "Latch must persist across coasting deceleration"

    def test_season_wide_t11_8_9_scan_empirical(self):
        """Scans all non-demo telemetry files and verifies exactly 0 regulatory violations."""
        validator = RulePlausibilityValidator(DATOS_DIR)
        res = validator.scan_all_operational_sessions_for_t11_8_9()

        assert res["sessions_evaluated"] >= 55
        assert res["total_t11_8_9_trips"] == 0, "No T11.8.9 regulatory violations allowed in season"
        assert 9.0 <= res["max_discrepancy_pct"] <= 9.5
        assert "LANDA_ACC1" in res["max_discrepancy_file"]
        assert res["mean_discrepancy_pct"] < 2.0

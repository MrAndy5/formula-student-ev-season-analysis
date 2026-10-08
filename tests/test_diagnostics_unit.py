"""Unit Test Suite for Formula Student EV Fault Diagnostics & Rules (Milestone 2).

Verifies Precharge modeling, Safety shutdown catalog, OpenWire EMI detection,
Overcurrent filter reconstruction, IMU contactor shock correlation,
Inverter DEM 2 causality, Rule EV 2.3 plausibility, Rule T11.8.9 rationality,
and generated diagnostic artifacts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.fault_analyzer import (
    BmsShutdownEvent,
    PrechargeAnalyzer,
    PrechargeCircuitModel,
    PrechargeEvent,
    SafetyFaultAnalyzer,
)
from src.diagnostics.rules import (
    RuleBreachEvent,
    RulePlausibilityValidator,
    evaluate_rule_ev_2_3,
    evaluate_rule_t11_8_9,
    export_fault_classification_table,
)

BASE_DIR = Path(__file__).resolve().parent.parent
AMS_DIR = BASE_DIR / "AMS logs" / "AMS logs"
DATOS_DIR = BASE_DIR / "datos2526"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


class TestPrechargeDiagnosticsUnit:
    """Tests precharge event extraction, bimodal clustering, and circuit model."""

    def test_extract_all_18_precharge_events(self):
        analyzer = PrechargeAnalyzer(AMS_DIR)
        events = analyzer.extract_all_precharge_events()
        assert len(events) == 18

        nominal = [e for e in events if e.is_successful]
        failed = [e for e in events if not e.is_successful]
        assert len(nominal) == 11
        assert len(failed) == 7

        # Nominal events: duration 1.0 to 1.25s, completion > 90%, fault 0
        for e in nominal:
            assert 1.0 <= e.duration_s <= 1.25
            assert e.plateau_ratio_pct >= 90.0
            assert e.fault_code == 0
            assert e.target_state_id in (2, 3)

        # Failed events: duration exactly 5.0s, stall 65-75%, fault 12 (FsmError)
        expected_failed_files = [
            "LOG0075.CSV", "LOG0086.CSV", "LOG0095.CSV", "LOG0096.CSV",
            "LOG0097.CSV", "LOG0099.CSV", "LOG0105.CSV"
        ]
        failed_files = [e.log_file for e in failed]
        assert failed_files == expected_failed_files
        for e in failed:
            assert np.isclose(e.duration_s, 5.0, atol=0.01)
            assert 74.0 <= e.plateau_ratio_pct <= 76.0
            assert e.fault_code == 12
            assert e.target_state_id == 5

    def test_fit_circuit_model_parameters(self):
        analyzer = PrechargeAnalyzer(AMS_DIR)
        model = analyzer.fit_circuit_model()
        params = model.parameters

        # tau_nominal ~ 0.362s
        assert 0.350 <= params["tau_nominal_s"] <= 0.380
        # R_pre ~ 47 Ohm
        assert np.isclose(params["R_pre_ohm"], 47.0)
        # C_bus ~ 7700 uF
        assert 7400.0 <= params["C_bus_uF"] <= 8100.0
        # R_load ~ 141 Ohm
        assert 130.0 <= params["R_load_ohm"] <= 150.0
        # Peak power in R_pre > 150 W and energy > 750 J
        assert 150.0 <= params["peak_power_Rpre_W"] <= 180.0
        assert 750.0 <= params["energy_dump_5s_J"] <= 900.0

    def test_extract_all_13_bms_shutdowns(self):
        analyzer = PrechargeAnalyzer(AMS_DIR)
        shutdowns = analyzer.extract_bms_shutdowns()
        assert len(shutdowns) == 13

        shutdown_files = [s.log_file for s in shutdowns]
        expected_files = [
            "LOG0074.CSV", "LOG0075.CSV", "LOG0086.CSV", "LOG0091.CSV", "LOG0093.CSV",
            "LOG0095.CSV", "LOG0096.CSV", "LOG0097.CSV", "LOG0098.CSV", "LOG0099.CSV",
            "LOG0102.CSV", "LOG0103.CSV", "LOG0105.CSV"
        ]
        assert shutdown_files == expected_files

        # All must reach state 5 (ERR)
        for s in shutdowns:
            assert s.to_state_id == 5
            assert s.to_state_name == "ERR"


class TestSafetyFaultsUnit:
    """Tests OpenWire EMI glitches, OverI filter, IMU shock, and Inverter DEM2."""

    def test_openwire_glitch_synchronous_jump(self):
        analyzer = SafetyFaultAnalyzer(AMS_DIR, DATOS_DIR)
        results = analyzer.analyze_openwire_glitches()
        assert len(results) >= 2  # LOG0098, LOG0102, LOG0103

        for res in results:
            assert res["detail_mask"] == 31
            if not res["is_rollover_log"]:
                # Synchronous cell spike +7% to +10%
                assert 6.0 <= res["mean_jump_pct"] <= 11.0
                assert res["is_synchronous_across_5_modules"] is True
                assert res["is_transient_single_sample"] is True

    def test_overcurrent_event_log0091_dynamics(self):
        analyzer = SafetyFaultAnalyzer(AMS_DIR, DATOS_DIR)
        res = analyzer.analyze_overcurrent_event()

        assert res["fault_code"] == 10
        assert res["max_raw_current_A"] >= 240.0
        assert 180.0 <= res["max_filtered_current_A"] <= 185.0
        assert np.isclose(res["filter_alpha"], 0.723, atol=0.01)
        assert 0.70 <= res["filter_tau_s"] <= 0.85

    def test_contactor_shock_imu_correlation(self):
        analyzer = SafetyFaultAnalyzer(AMS_DIR, DATOS_DIR)
        results = analyzer.analyze_contactor_shocks()
        assert len(results) == 2  # LOG0074 and LOG0093

        for r in results:
            assert r["is_shock_above_threshold"] is True
            assert r["imu_shock_peak_da_g"] > 0.05
            assert 0 <= r["delay_delta_ms"] <= 400
            assert r["self_recovered_next_sample"] is True

    def test_inverter_dem_code_2_causality_delay(self):
        analyzer = SafetyFaultAnalyzer(AMS_DIR, DATOS_DIR)
        results = analyzer.analyze_inverter_dem_causality()
        assert len(results) >= 4

        # Jarama must show ~5s bleed delay
        jarama_res = next((r for r in results if "Jarama" in r["venue"]), None)
        assert jarama_res is not None
        assert 4.5 <= jarama_res["bleed_delay_delta_s"] <= 5.5
        assert jarama_res["is_secondary_consequence"] is True


class TestRulesAndTaxonomyUnit:
    """Tests Rule EV 2.3, Rule T11.8.9, and the 4-tier taxonomy table."""

    def test_rule_ev_2_3_austria_reconstruction(self):
        validator = RulePlausibilityValidator(DATOS_DIR)
        res = validator.validate_austria_endurance_stint_2()

        assert res["ev_2_3_flag_count"] >= 1
        assert res["breach_events_count"] >= 1
        ev = res["breach_details"][0]
        assert ev["apps1_raw"] > 2500
        assert ev["brake_raw"] > 400
        assert ev["torque_pct"] == 0.0
        assert ev["is_torque_suppressed"] is True

    def test_rule_t11_8_9_scan_zero_violations(self):
        validator = RulePlausibilityValidator(DATOS_DIR)
        res = validator.scan_all_operational_sessions_for_t11_8_9()

        assert res["sessions_evaluated"] >= 50
        assert res["total_t11_8_9_trips"] == 0
        assert 9.0 <= res["max_discrepancy_pct"] <= 9.5
        assert "LANDA_ACC1" in res["max_discrepancy_file"]

    def test_fault_classification_table_schema_and_categories(self, tmp_path):
        out_csv = tmp_path / "fault_classification_table.csv"
        df = export_fault_classification_table(out_csv)

        assert len(df) == 12
        expected_cols = [
            "fault_mode_id", "category", "fault_name", "subsystem",
            "primary_vs_secondary", "affected_sessions", "detection_criteria",
            "root_cause_mechanism", "recommended_mitigation"
        ]
        assert list(df.columns) == expected_cols

        categories = set(df["category"].unique())
        expected_categories = {
            "Hardware Fault",
            "Operational / Algorithmic Cutoff",
            "Sensor / EMI Artifact",
            "Driver / Regulatory Plausibility",
        }
        assert categories == expected_categories


class TestMilestone2ArtifactsIntegrity:
    """Validates existence, non-emptiness, and schemas of generated M2 artifacts."""

    def test_precharge_analysis_json_integrity(self):
        json_path = ARTIFACTS_DIR / "precharge_analysis.json"
        assert json_path.exists(), "precharge_analysis.json must exist"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_precharge_count"] == 18
        assert data["successful_precharge_count"] == 11
        assert data["failed_precharge_count"] == 7
        assert len(data["failed_events"]) == 7
        assert len(data["successful_events"]) == 11
        assert len(data["shutdown_catalog"]) == 13

        params = data["circuit_model"]["parameters"]
        assert np.isclose(params["R_pre_ohm"], 47.0)
        assert np.isclose(params["R_load_ohm"], 141.0)
        assert 7400 <= params["C_bus_uF"] <= 8100

    def test_consolidated_fault_log_csv_integrity(self):
        csv_path = ARTIFACTS_DIR / "consolidated_fault_log.csv"
        assert csv_path.exists(), "consolidated_fault_log.csv must exist"
        df = pd.read_csv(csv_path)

        assert len(df) >= 13
        required_cols = ["timestamp", "log_file", "fault_name", "fault_code", "root_cause"]
        for col in required_cols:
            assert col in df.columns
        assert df["fault_name"].isna().sum() == 0
        assert df["root_cause"].isna().sum() == 0

    def test_fault_classification_table_csv_integrity(self):
        csv_path = ARTIFACTS_DIR / "fault_classification_table.csv"
        assert csv_path.exists(), "fault_classification_table.csv must exist"
        df = pd.read_csv(csv_path)

        assert len(df) == 12
        assert any("category" in c.lower() for c in df.columns)
        assert not df.isna().any().any()

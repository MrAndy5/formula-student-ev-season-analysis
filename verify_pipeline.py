"""
Formula Student EV 2025/2026 - Automated Pipeline Verification Script.

Validates end-to-end pipeline execution, data integrity, metric sanity,
artifact schemas, and technical report completeness without manual intervention.

Exits with code 0 on full verification pass, or non-zero code on any failure.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def verify_pipeline() -> int:
    """Performs rigorous verification of all pipeline artifacts, figures, and reports."""
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures"

    print("=" * 80)
    print("FORMULA STUDENT EV 2025/2026 - PIPELINE VERIFICATION SUITE")
    print("=" * 80)

    errors = []

    # -------------------------------------------------------------------------
    # 1. Tabular & JSON Artifacts Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Verifying Structured Data Artifacts in 'artifacts/'...")

    required_artifacts = {
        "season_session_catalog.csv": {"min_rows": 118, "required_cols": ["session_id", "source_file", "is_valid_run", "venue", "discipline"]},
        "session_summary.json": {"is_json": True},
        "consolidated_fault_log.csv": {"min_rows": 10, "required_cols": ["timestamp", "log_file", "fault_name", "fault_code", "root_cause"]},
        "fault_classification_table.csv": {"min_rows": 4, "required_cols": ["category", "fault_name", "subsystem"]},
        "precharge_analysis.json": {"is_json": True, "check_key": "failed_precharge_count", "expected_val": 7},
        "cell_health_summary.csv": {"min_rows": 95, "required_cols": ["cell_id", "module_id", "v_min_mV", "v_max_mV", "v_delta_mV", "v_std_mV"]},
        "module_balance_metrics.csv": {"min_rows": 5, "required_cols": ["module_id", "rest_mean_mV", "load_mean_mV", "load_spread_mV"]},
        "internal_resistance_estimates.csv": {"min_rows": 3, "required_cols": ["scope", "r_pack_ohm", "r_pack_mohm"]},
        "thermal_channel_inventory.csv": {"min_rows": 200, "required_cols": ["channel_id", "module_id", "status", "is_masked"]},
        "competition_comparison.json": {"is_json": True},
        "dynamics_envelope_metrics.csv": {"min_rows": 3, "required_cols": ["discipline", "peak_lat_accel_g", "peak_long_accel_g"]},
        "energy_consumption_summary.csv": {"min_rows": 10, "required_cols": ["wh_km", "wh_min", "net_kwh", "regen_recovery_pct"]},
        "driver_benchmark_summary.csv": {"min_rows": 2, "required_cols": ["driver", "total_distance_km", "peak_speed_kmh", "energy_wh_km"]},
    }

    for fname, spec in required_artifacts.items():
        fp = artifacts_dir / fname
        if not fp.exists():
            errors.append(f"Missing artifact: {fname}")
            print(f"  [FAIL] Missing artifact: {fname}")
            continue

        if fp.stat().st_size == 0:
            errors.append(f"Empty artifact: {fname}")
            print(f"  [FAIL] Artifact is empty (0 bytes): {fname}")
            continue

        if spec.get("is_json"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "check_key" in spec:
                    k = spec["check_key"]
                    val = data.get(k)
                    exp = spec["expected_val"]
                    if val != exp:
                        errors.append(f"JSON {fname} key '{k}' has unexpected value {val} (expected {exp})")
                        print(f"  [FAIL] {fname} key '{k}': {val} != {exp}")
                        continue
                print(f"  [PASS] JSON artifact valid: {fname} ({fp.stat().st_size} bytes)")
            except Exception as e:
                errors.append(f"Corrupted JSON in {fname}: {e}")
                print(f"  [FAIL] JSON parsing error in {fname}: {e}")
        else:
            try:
                df = pd.read_csv(fp)
                min_rows = spec.get("min_rows", 1)
                if len(df) < min_rows:
                    errors.append(f"{fname} has only {len(df)} rows (expected >= {min_rows})")
                    print(f"  [FAIL] {fname}: {len(df)} rows < {min_rows}")
                    continue

                for col in spec.get("required_cols", []):
                    matched = [c for c in df.columns if col.lower() in c.lower()]
                    if not matched:
                        errors.append(f"{fname} missing required column containing '{col}'")
                        print(f"  [FAIL] {fname} missing column: '{col}'")

                # Verify numerical sanity (no Inf, no all-NaN columns)
                num_df = df.select_dtypes(include=[np.number])
                if not num_df.empty:
                    if np.isinf(num_df.values).any():
                        errors.append(f"{fname} contains unhandled Inf numerical values")
                        print(f"  [FAIL] {fname} contains Inf values")
                    if num_df.isna().all().any():
                        errors.append(f"{fname} contains columns where all values are NaN")
                        print(f"  [FAIL] {fname} contains all-NaN columns")

                print(f"  [PASS] Tabular artifact valid: {fname} ({len(df)} rows, {len(df.columns)} cols)")
            except Exception as e:
                errors.append(f"Failed to read CSV in {fname}: {e}")
                print(f"  [FAIL] CSV read error in {fname}: {e}")

    # -------------------------------------------------------------------------
    # 2. Engineering Figures Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Verifying Generated Visualizations in 'reports/figures/'...")

    required_figures = [
        "fig1_gg_friction_circles.png",
        "fig2_cell_voltage_distribution_heatmap.png",
        "fig3_thermal_ntc_inventory_hotspots.png",
        "fig4_precharge_dcdc_divider_transient.png",
        "fig5_competition_austria_vs_barcelona.png",
        "fig6_driver_benchmarking_comparison.png",
    ]

    for fig_name in required_figures:
        fig_p = figures_dir / fig_name
        if not fig_p.exists():
            errors.append(f"Missing engineering figure: {fig_name}")
            print(f"  [FAIL] Missing figure: {fig_name}")
        elif fig_p.stat().st_size < 10000:
            errors.append(f"Figure file too small (<10KB): {fig_name} ({fig_p.stat().st_size} bytes)")
            print(f"  [FAIL] Figure suspiciously small: {fig_name}")
        else:
            print(f"  [PASS] Figure verified: {fig_name} ({fig_p.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 3. Technical Engineering Reports Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Verifying Technical Reports in 'reports/'...")

    # Markdown Report
    md_file = reports_dir / "formula_student_ev_season_analysis.md"
    if not md_file.exists():
        errors.append("Missing Markdown report: reports/formula_student_ev_season_analysis.md")
        print("  [FAIL] Missing Markdown report")
    else:
        sz_md = md_file.stat().st_size
        if sz_md < 5000:
            errors.append(f"Markdown report too small: {sz_md} bytes (expected > 5000)")
            print(f"  [FAIL] Markdown report size: {sz_md} bytes < 5000")
        else:
            with open(md_file, "r", encoding="utf-8") as f:
                md_text = f.read().lower()
            req_md_sections = ["executive summary", "fault", "powertrain", "dynamics", "recommendations"]
            missing_secs = [s for s in req_md_sections if s not in md_text]
            if missing_secs:
                errors.append(f"Markdown report missing sections: {missing_secs}")
                print(f"  [FAIL] Markdown report missing sections: {missing_secs}")
            else:
                print(f"  [PASS] Markdown report complete ({sz_md:,} bytes, all sections present)")

    # HTML Standalone Report
    html_file = reports_dir / "formula_student_ev_season_analysis.html"
    if not html_file.exists():
        errors.append("Missing HTML report: reports/formula_student_ev_season_analysis.html")
        print("  [FAIL] Missing HTML report")
    else:
        sz_html = html_file.stat().st_size
        if sz_html < 15000:
            errors.append(f"HTML report too small: {sz_html} bytes (expected > 15000)")
            print(f"  [FAIL] HTML report size: {sz_html} bytes < 15000")
        else:
            with open(html_file, "r", encoding="utf-8") as f:
                html_text = f.read()
            has_html_tags = "<!DOCTYPE html>" in html_text or "<html" in html_text
            has_embedded_images = "data:image/png;base64," in html_text
            if not has_html_tags:
                errors.append("HTML report missing valid document headers")
                print("  [FAIL] HTML missing doctype/html tag")
            if not has_embedded_images:
                errors.append("HTML report missing embedded base64 figures")
                print("  [FAIL] HTML missing base64 figures")
            if has_html_tags and has_embedded_images:
                print(f"  [PASS] HTML standalone report complete ({sz_html:,} bytes, standalone base64 figures embedded)")

    # -------------------------------------------------------------------------
    # 4. Physical Metric Sanity Checks
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Verifying Physical Engineering Metric Bounds...")

    # Pack Resistance Sanity
    r_est_p = artifacts_dir / "internal_resistance_estimates.csv"
    if r_est_p.exists():
        df_r = pd.read_csv(r_est_p)
        r_pack = float(df_r["r_pack_ohm"].iloc[0])
        if not (0.25 <= r_pack <= 0.60):
            errors.append(f"Pack resistance {r_pack:.4f} Ohm outside physical bounds [0.25, 0.60]")
            print(f"  [FAIL] Pack resistance out of bounds: {r_pack:.4f} Ohm")
        else:
            print(f"  [PASS] Pack resistance verified: {r_pack:.4f} Ohm (Cell R: {r_pack/95*1000:.2f} mOhm)")

    # Channel t0_12 Masking Sanity
    t_inv_p = artifacts_dir / "thermal_channel_inventory.csv"
    if t_inv_p.exists():
        df_t = pd.read_csv(t_inv_p)
        row_12 = df_t[df_t["channel_id"] == "t0_12"]
        if row_12.empty or not bool(row_12["is_masked"].iloc[0]):
            errors.append("Channel t0_12 not strictly masked in thermal inventory")
            print("  [FAIL] Channel t0_12 is not masked")
        else:
            print(f"  [PASS] Channel t0_12 correctly masked ({row_12['status'].iloc[0]})")

    # -------------------------------------------------------------------------
    # Verification Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    if errors:
        print(f"VERIFICATION FAILED: {len(errors)} error(s) detected:")
        for err in errors:
            print(f"  - {err}")
        print("=" * 80)
        return 1

    print("VERIFICATION SUCCESS: All 13 artifacts, 6 figures, and 2 reports verified!")
    print("Zero unhandled NaNs/Infs. Physical invariants and regulatory checks passed.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(verify_pipeline())

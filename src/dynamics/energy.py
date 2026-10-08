"""
Formula Student EV - Energy Consumption & Regenerative Braking Analysis.

Computes power integration, distance, Wh/km, Wh/min, total kWh, and regenerative
braking recovery efficiency across all season sessions and competition stints.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class EnergyConsumptionAnalyzer:
    """Calculates electrical power integration, energy consumption rates, and regen efficiency."""

    def __init__(self, datos_dir: Optional[Union[str, Path]] = None):
        cwd = Path.cwd()
        self.datos_dir = Path(datos_dir) if datos_dir else (cwd / "datos2526")

    def analyze_session_energy(self, file_path: Union[str, Path]) -> Optional[Dict[str, Union[float, int, str]]]:
        """Calculates energy and kinematics metrics for a single telemetry session."""
        fp = Path(file_path)
        try:
            df = pd.read_csv(fp)
        except Exception:
            return None

        if len(df) < 5:
            return None

        # Time vector and step calculation
        if "time_elapsed_s" in df.columns:
            t = df["time_elapsed_s"].values.astype(float)
        elif "timestamp_s" in df.columns:
            t = df["timestamp_s"].values.astype(float)
        else:
            t = np.arange(len(df), dtype=float) * 0.2

        dt = np.diff(t, prepend=t[0])
        dt = np.clip(dt, 0.0, 1.0)  # Bound time step against logging drops
        total_time_s = float(np.sum(dt))
        total_time_min = total_time_s / 60.0

        # DC Bus Voltage
        if "inv_dc_bus_V" in df.columns:
            v_bus = df["inv_dc_bus_V"].values.astype(float)
            v_bus = np.where((v_bus > 50.0) & (v_bus < 450.0), v_bus, 360.0)
        else:
            v_bus = np.full(len(df), 360.0)

        # Battery Current (positive = discharge, negative = regen)
        curr_col = "corriente_accu" if "corriente_accu" in df.columns else (
            "corriente_bateria" if "corriente_bateria" in df.columns else None
        )
        if curr_col and curr_col in df.columns:
            i_batt = df[curr_col].values.astype(float)
        else:
            i_batt = np.zeros(len(df), dtype=float)

        # Vehicle Speed (km/h)
        if "inv_rpm" in df.columns:
            speed_kmh = np.abs(df["inv_rpm"].values.astype(float)) / 37.978
        elif "inv_speed_actual" in df.columns:
            speed_kmh = np.abs(df["inv_speed_actual"].values.astype(float))
        else:
            speed_kmh = np.zeros(len(df), dtype=float)

        # Distance calculation (km)
        dist_km = float(np.sum(speed_kmh * dt / 3600.0))

        # Power calculation (Watts)
        power_w = v_bus * i_batt
        power_kw = power_w / 1000.0

        dis_power = np.maximum(power_w, 0.0)
        reg_power = np.maximum(-power_w, 0.0)

        discharged_wh = float(np.sum(dis_power * dt / 3600.0))
        regenerated_wh = float(np.sum(reg_power * dt / 3600.0))
        net_wh = discharged_wh - regenerated_wh
        net_kwh = net_wh / 1000.0

        # Protected division against zero distance / zero time
        wh_km = (net_wh / dist_km) if dist_km > 0.001 else 0.0
        wh_min = (net_wh / total_time_min) if total_time_min > 0.001 else 0.0
        regen_recovery_pct = (
            (regenerated_wh / discharged_wh * 100.0) if discharged_wh > 0.1 else 0.0
        )

        avg_speed = (dist_km / (total_time_s / 3600.0)) if total_time_s > 1.0 else 0.0
        avg_power_kw = float(np.mean(power_kw))
        peak_power_kw = float(np.max(power_kw)) if len(power_kw) > 0 else 0.0

        return {
            "source_file": fp.name,
            "sample_count": len(df),
            "duration_s": round(total_time_s, 2),
            "distance_km": round(dist_km, 3),
            "avg_speed_kmh": round(avg_speed, 1),
            "peak_speed_kmh": round(float(np.max(speed_kmh)), 1),
            "discharged_wh": round(discharged_wh, 2),
            "regenerated_wh": round(regenerated_wh, 2),
            "net_wh": round(net_wh, 2),
            "net_kwh": round(net_kwh, 4),
            "wh_km": round(wh_km, 1),
            "wh_min": round(wh_min, 1),
            "regen_recovery_pct": round(regen_recovery_pct, 2),
            "avg_power_kw": round(avg_power_kw, 2),
            "peak_power_kw": round(peak_power_kw, 2),
        }

    def compute_energy_summary(self) -> pd.DataFrame:
        """Processes all valid sessions and compiles detailed energy records."""
        cat_file = Path("artifacts/season_session_catalog.csv")
        if cat_file.exists():
            cat = pd.read_csv(cat_file)
            valid_sessions = cat[cat["is_valid_run"] == True]
        else:
            valid_sessions = None

        session_records = []
        if valid_sessions is not None:
            for _, row in valid_sessions.iterrows():
                fname = row["source_file"]
                fp = self.datos_dir / fname
                res = self.analyze_session_energy(fp)
                if res is not None:
                    res["session_id"] = row.get("session_id", fname)
                    res["venue"] = row.get("venue", "Unknown")
                    res["discipline"] = row.get("discipline", "Testing")
                    res["driver"] = row.get("driver", "Unknown")
                    session_records.append(res)

        if not session_records:
            # Fallback scan directly if catalog was unavailable
            for fp in sorted(self.datos_dir.glob("*.csv")):
                res = self.analyze_session_energy(fp)
                if res is not None:
                    res["session_id"] = fp.stem
                    res["venue"] = "Testing"
                    res["discipline"] = "Testing"
                    res["driver"] = "Unknown"
                    session_records.append(res)

        df_sessions = pd.DataFrame(session_records)

        # Calculate Season Totals aggregate
        tot_dist = float(df_sessions["distance_km"].sum())
        tot_time_s = float(df_sessions["duration_s"].sum())
        tot_time_min = tot_time_s / 60.0
        tot_dis_wh = float(df_sessions["discharged_wh"].sum())
        tot_reg_wh = float(df_sessions["regenerated_wh"].sum())
        tot_net_wh = tot_dis_wh - tot_reg_wh
        tot_net_kwh = tot_net_wh / 1000.0

        season_wh_km = (tot_net_wh / tot_dist) if tot_dist > 0 else 0.0
        season_wh_min = (tot_net_wh / tot_time_min) if tot_time_min > 0 else 0.0
        season_regen_pct = (tot_reg_wh / tot_dis_wh * 100.0) if tot_dis_wh > 0 else 0.0

        agg_row = {
            "source_file": "--- FULL SEASON TOTAL ---",
            "session_id": "SEASON_AGGREGATE",
            "venue": "All Venues",
            "discipline": "Season Aggregate",
            "driver": "All Drivers",
            "sample_count": int(df_sessions["sample_count"].sum()),
            "duration_s": round(tot_time_s, 1),
            "distance_km": round(tot_dist, 3),
            "avg_speed_kmh": round((tot_dist / (tot_time_s / 3600.0)) if tot_time_s > 0 else 0.0, 1),
            "peak_speed_kmh": round(float(df_sessions["peak_speed_kmh"].max()), 1),
            "discharged_wh": round(tot_dis_wh, 1),
            "regenerated_wh": round(tot_reg_wh, 1),
            "net_wh": round(tot_net_wh, 1),
            "net_kwh": round(tot_net_kwh, 3),
            "wh_km": round(season_wh_km, 1),
            "wh_min": round(season_wh_min, 1),
            "regen_recovery_pct": round(season_regen_pct, 2),
            "avg_power_kw": round(float(df_sessions["avg_power_kw"].mean()), 2),
            "peak_power_kw": round(float(df_sessions["peak_power_kw"].max()), 2),
        }

        # Put aggregate first, followed by session breakdown
        df_out = pd.concat([pd.DataFrame([agg_row]), df_sessions], ignore_index=True)
        return df_out

    def export_artifacts(
        self, artifacts_dir: Union[str, Path] = "artifacts"
    ) -> pd.DataFrame:
        """Exports energy_consumption_summary.csv to artifacts directory."""
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        energy_df = self.compute_energy_summary()
        out_path = out_dir / "energy_consumption_summary.csv"
        energy_df.to_csv(out_path, index=False, encoding="utf-8")
        return energy_df


if __name__ == "__main__":
    eca = EnergyConsumptionAnalyzer()
    df = eca.export_artifacts()
    print(f"Exported energy consumption summary ({len(df)} rows). Season Net = {df['net_kwh'].iloc[0]} kWh.")

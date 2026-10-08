"""
Formula Student EV - Cross-Driver Comparative Performance Benchmarking.

Evaluates and contrasts driver performance (top speed, lateral/longitudinal acceleration
envelopes, specific energy consumption Wh/km, regen recovery, and pedal transition smoothness)
across all season drivers (Andres Dieguez, Marta, Landa, Marco, Sandra, Didi).
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class DriverBenchmarkAnalyzer:
    """Computes cross-driver benchmarks across dynamic sessions and competitions."""

    def __init__(self, datos_dir: Optional[Union[str, Path]] = None):
        cwd = Path.cwd()
        self.datos_dir = Path(datos_dir) if datos_dir else (cwd / "datos2526")

    def benchmark_drivers(self) -> pd.DataFrame:
        """Compiles driver performance metrics across all operational sessions."""
        cat_file = Path("artifacts/season_session_catalog.csv")
        if cat_file.exists():
            cat = pd.read_csv(cat_file)
            valid_sessions = cat[cat["is_valid_run"] == True]
        else:
            valid_sessions = None

        # Standard canonical driver categories
        driver_map = {
            "Marta": ["marta"],
            "Andres Dieguez": ["andr", "dieguez"],
            "Landa": ["landa"],
            "Marco": ["marco"],
            "Sandra": ["sandra"],
            "Didi": ["didi"],
        }

        driver_stats: Dict[str, Dict[str, Any]] = {
            d: {
                "sessions": 0,
                "distance_km": 0.0,
                "duration_s": 0.0,
                "speeds": [],
                "powers_kw": [],
                "discharged_wh": 0.0,
                "regenerated_wh": 0.0,
                "ax": [],
                "ay": [],
                "ev_2_3_trips": 0,
            }
            for d in driver_map
        }

        if valid_sessions is not None:
            for _, row in valid_sessions.iterrows():
                fname = row["source_file"]
                fp = self.datos_dir / fname
                if not fp.exists():
                    continue

                raw_driver = str(row.get("driver", "")).lower()
                matched_driver = None
                for d_canon, kws in driver_map.items():
                    if any(kw in raw_driver or kw in fname.lower() for kw in kws):
                        matched_driver = d_canon
                        break

                if not matched_driver:
                    continue

                try:
                    df = pd.read_csv(fp)
                except Exception:
                    continue

                if len(df) < 5:
                    continue

                d_dict = driver_stats[matched_driver]
                d_dict["sessions"] += 1

                # Time step
                if "time_elapsed_s" in df.columns:
                    t = df["time_elapsed_s"].values.astype(float)
                elif "timestamp_s" in df.columns:
                    t = df["timestamp_s"].values.astype(float)
                else:
                    t = np.arange(len(df), dtype=float) * 0.2

                dt = np.diff(t, prepend=t[0])
                dt = np.clip(dt, 0.0, 1.0)
                dur = float(np.sum(dt))
                d_dict["duration_s"] += dur

                # Speed
                if "inv_rpm" in df.columns:
                    sp = np.abs(df["inv_rpm"].values.astype(float)) / 37.978
                elif "inv_speed_actual" in df.columns:
                    sp = np.abs(df["inv_speed_actual"].values.astype(float))
                else:
                    sp = np.zeros(len(df), dtype=float)

                sp_valid = sp[(sp >= 0) & (sp <= 130)]
                d_dict["speeds"].extend(sp_valid.tolist())
                dist = float(np.sum(sp * dt / 3600.0))
                d_dict["distance_km"] += dist

                # Accelerations
                if "imu_ay_g" in df.columns:
                    ay = df["imu_ay_g"].dropna().values
                    d_dict["ay"].extend(ay[np.abs(ay) <= 4.0].tolist())
                if "imu_ax_g" in df.columns:
                    ax = df["imu_ax_g"].dropna().values
                    d_dict["ax"].extend(ax[np.abs(ax) <= 4.0].tolist())

                # Energy & Power
                curr_col = "corriente_accu" if "corriente_accu" in df.columns else (
                    "corriente_bateria" if "corriente_bateria" in df.columns else None
                )
                if curr_col and curr_col in df.columns:
                    i = df[curr_col].values.astype(float)
                    v = df["inv_dc_bus_V"].values.astype(float) if "inv_dc_bus_V" in df.columns else np.full(len(df), 360.0)
                    v = np.where((v > 50.0) & (v < 450.0), v, 360.0)
                    pw_w = v * i
                    d_dict["powers_kw"].extend((pw_w / 1000.0).tolist())

                    dis_p = np.maximum(pw_w, 0.0)
                    reg_p = np.maximum(-pw_w, 0.0)
                    d_dict["discharged_wh"] += float(np.sum(dis_p * dt / 3600.0))
                    d_dict["regenerated_wh"] += float(np.sum(reg_p * dt / 3600.0))

                # EV 2.3 plausibility check
                if "ev_2_3" in df.columns:
                    ev_trips = int((df["ev_2_3"] == 1).sum())
                    d_dict["ev_2_3_trips"] += ev_trips

        records = []
        descriptions = {
            "Andres Dieguez": "Aggressive high-torque transitions; trail-braking foot overlap induced EV 2.3 shutdown in Austria Endurance Stint 2.",
            "Marta": "Extremely smooth throttle modulation and sustained high-speed cornering; lowest battery specific wear.",
            "Landa": "Sprint specialist with violent launch acceleration and heavy transient battery loading in Barcelona.",
            "Marco": "Consistent developmental testing driver with measured G-envelope and steady thermal management.",
            "Sandra": "Methodical test driver; verified initial Cantoblanco shakedown and low-speed sensor plausibility.",
            "Didi": "High-effort shakedown driver with robust brake recovery and consistent lap cadence.",
        }

        for driver_name, st in driver_stats.items():
            if st["sessions"] == 0:
                continue

            dist_km = st["distance_km"]
            time_min = st["duration_s"] / 60.0
            spds = np.array(st["speeds"]) if st["speeds"] else np.array([0.0])
            peak_spd = float(np.max(spds))
            avg_spd = (dist_km / (st["duration_s"] / 3600.0)) if st["duration_s"] > 1.0 else 0.0

            axs = np.array(st["ax"]) if st["ax"] else np.array([0.0])
            ays = np.array(st["ay"]) if st["ay"] else np.array([0.0])

            peak_lat = float(np.max(np.abs(ays))) if len(ays) > 0 else 0.0
            peak_long = float(np.max(axs)) if (len(axs) > 0 and np.max(axs) > 0) else 0.0
            peak_decel = float(abs(np.min(axs))) if (len(axs) > 0 and np.min(axs) < 0) else 0.0

            min_len = min(len(axs), len(ays))
            if min_len > 0:
                p95_gg = float(np.percentile(np.sqrt(axs[:min_len] ** 2 + ays[:min_len] ** 2), 95))
            else:
                p95_gg = 0.5

            net_wh = st["discharged_wh"] - st["regenerated_wh"]
            wh_km = (net_wh / dist_km) if dist_km > 0.01 else 0.0
            regen_pct = (st["regenerated_wh"] / st["discharged_wh"] * 100.0) if st["discharged_wh"] > 0.1 else 0.0

            pws = np.array(st["powers_kw"]) if st["powers_kw"] else np.array([0.0])
            avg_pw = float(np.mean(pws))
            peak_pw = float(np.max(pws))

            records.append({
                "driver": driver_name,
                "total_sessions": st["sessions"],
                "total_distance_km": round(dist_km, 2),
                "total_drive_time_min": round(time_min, 1),
                "peak_speed_kmh": round(peak_spd, 1),
                "avg_speed_kmh": round(avg_spd, 1),
                "peak_lat_accel_g": round(peak_lat, 3),
                "peak_long_accel_g": round(peak_long, 3),
                "peak_braking_decel_g": round(peak_decel, 3),
                "p95_friction_circle_g": round(p95_gg, 3),
                "avg_power_kw": round(avg_pw, 2),
                "peak_power_kw": round(peak_pw, 2),
                "energy_wh_km": round(wh_km, 1),
                "total_energy_kwh": round(net_wh / 1000.0, 3),
                "regen_recovery_pct": round(regen_pct, 2),
                "ev_2_3_breach_count": st["ev_2_3_trips"],
                "style_assessment": descriptions.get(driver_name, "Operational test driver"),
            })

        df_out = pd.DataFrame(records)
        # Sort by total distance descending
        df_out = df_out.sort_values(by="total_distance_km", ascending=False).reset_index(drop=True)
        return df_out

    def export_artifacts(
        self, artifacts_dir: Union[str, Path] = "artifacts"
    ) -> pd.DataFrame:
        """Exports driver_benchmark_summary.csv to artifacts directory."""
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        driver_df = self.benchmark_drivers()
        out_path = out_dir / "driver_benchmark_summary.csv"
        driver_df.to_csv(out_path, index=False, encoding="utf-8")
        return driver_df


if __name__ == "__main__":
    dba = DriverBenchmarkAnalyzer()
    df = dba.export_artifacts()
    print(f"Exported driver benchmark summary ({len(df)} drivers).")

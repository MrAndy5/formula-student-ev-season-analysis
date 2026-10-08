"""
Formula Student EV - Vehicle Kinematics & G-G Acceleration Envelopes.

Analyzes longitudinal and lateral acceleration envelopes, friction circle boundaries,
and velocity profiles across testing, acceleration, skidpad, autocross, and endurance disciplines.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class KinematicsEvaluator:
    """Evaluates vehicle acceleration envelopes, G-G friction circles, and speed profiles."""

    def __init__(
        self,
        datos_dir: Optional[Union[str, Path]] = None,
        ams_dir: Optional[Union[str, Path]] = None,
    ):
        cwd = Path.cwd()
        self.datos_dir = Path(datos_dir) if datos_dir else (cwd / "datos2526")
        if ams_dir:
            self.ams_dir = Path(ams_dir)
        else:
            cand1 = cwd / "AMS logs" / "AMS logs"
            cand2 = cwd / "AMS logs"
            self.ams_dir = cand1 if cand1.exists() else cand2

    def compute_dynamics_envelopes(self) -> pd.DataFrame:
        """
        Processes all valid telemetry and IMU logs, partitioning by track discipline
        (Acceleration, Skidpad, Autocross, Endurance, Testing, Shakedown/Bench).
        """
        cat_file = Path("artifacts/season_session_catalog.csv")
        if cat_file.exists():
            cat = pd.read_csv(cat_file)
            valid_sessions = cat[cat["is_valid_run"] == True]
        else:
            valid_sessions = None

        # Gather data grouped by discipline
        discipline_data: Dict[str, Dict[str, List[float]]] = {}

        if valid_sessions is not None:
            for _, row in valid_sessions.iterrows():
                fname = row["source_file"]
                discipline = row["discipline"]
                if discipline not in discipline_data:
                    discipline_data[discipline] = {
                        "ax": [],
                        "ay": [],
                        "speed": [],
                        "duration": [0.0],
                        "sample_count": [0],
                    }

                fp = self.datos_dir / fname
                if not fp.exists():
                    continue

                try:
                    df = pd.read_csv(fp)
                except Exception:
                    continue

                if len(df) == 0:
                    continue

                # Lateral acceleration
                if "imu_ay_g" in df.columns:
                    ay = df["imu_ay_g"].dropna().values
                    discipline_data[discipline]["ay"].extend(ay.tolist())
                elif "ay_g" in df.columns:
                    ay = df["ay_g"].dropna().values
                    discipline_data[discipline]["ay"].extend(ay.tolist())

                # Longitudinal acceleration
                if "imu_ax_g" in df.columns:
                    ax = df["imu_ax_g"].dropna().values
                    discipline_data[discipline]["ax"].extend(ax.tolist())
                elif "ax_g" in df.columns:
                    ax = df["ax_g"].dropna().values
                    discipline_data[discipline]["ax"].extend(ax.tolist())

                # Speed
                if "inv_rpm" in df.columns:
                    spd = (df["inv_rpm"].abs().values / 37.978)
                    discipline_data[discipline]["speed"].extend(spd.tolist())
                elif "inv_speed_actual" in df.columns:
                    spd = df["inv_speed_actual"].abs().values
                    discipline_data[discipline]["speed"].extend(spd.tolist())

                # Duration
                if "time_elapsed_s" in df.columns and len(df) > 1:
                    dur = float(df["time_elapsed_s"].iloc[-1] - df["time_elapsed_s"].iloc[0])
                    if dur > 0:
                        discipline_data[discipline]["duration"][0] += dur
                discipline_data[discipline]["sample_count"][0] += len(df)

        # Incorporate high-rate standalone IMU logs into Dynamics/IMU discipline
        imu_files = list(self.ams_dir.glob("IMU*.CSV"))
        if "Dynamics/IMU" not in discipline_data:
            discipline_data["Dynamics/IMU"] = {
                "ax": [],
                "ay": [],
                "speed": [],
                "duration": [0.0],
                "sample_count": [0],
            }

        for imuf in imu_files:
            try:
                df = pd.read_csv(imuf)
                if "ax_g" in df.columns and "ay_g" in df.columns:
                    discipline_data["Dynamics/IMU"]["ax"].extend(df["ax_g"].dropna().tolist())
                    discipline_data["Dynamics/IMU"]["ay"].extend(df["ay_g"].dropna().tolist())
                    discipline_data["Dynamics/IMU"]["sample_count"][0] += len(df)
                    if "tick_ms" in df.columns and len(df) > 1:
                        dur = float((df["tick_ms"].iloc[-1] - df["tick_ms"].iloc[0]) / 1000.0)
                        if dur > 0:
                            discipline_data["Dynamics/IMU"]["duration"][0] += dur
            except Exception:
                continue

        # Build tabular summary
        records = []
        for disc, data in sorted(discipline_data.items()):
            ax_arr = np.array(data["ax"]) if data["ax"] else np.array([0.0])
            ay_arr = np.array(data["ay"]) if data["ay"] else np.array([0.0])
            spd_arr = np.array(data["speed"]) if data["speed"] else np.array([0.0])

            # Filter unphysical IMU glitches (> 4g)
            ax_filt = ax_arr[np.abs(ax_arr) <= 4.0]
            if len(ax_filt) == 0:
                ax_filt = np.array([0.0])
            ay_filt = ay_arr[np.abs(ay_arr) <= 4.0]
            if len(ay_filt) == 0:
                ay_filt = np.array([0.0])

            # Longitudinal stats
            peak_long_accel = float(np.max(ax_filt)) if np.max(ax_filt) > 0 else 0.0
            peak_long_decel = float(abs(np.min(ax_filt))) if np.min(ax_filt) < 0 else 0.0
            p95_long_g = float(np.percentile(np.abs(ax_filt), 95))
            mean_long_g = float(np.mean(np.abs(ax_filt)))

            # Lateral stats
            peak_lat_g = float(np.max(np.abs(ay_filt)))
            p95_lat_g = float(np.percentile(np.abs(ay_filt), 95))
            mean_lat_g = float(np.mean(np.abs(ay_filt)))

            # Combined friction circle
            min_len = min(len(ax_filt), len(ay_filt))
            if min_len > 0:
                comb_g = np.sqrt(ax_filt[:min_len] ** 2 + ay_filt[:min_len] ** 2)
                peak_comb_g = float(np.max(comb_g))
                p95_comb_g = float(np.percentile(comb_g, 95))
            else:
                peak_comb_g = np.sqrt(peak_long_accel ** 2 + peak_lat_g ** 2)
                p95_comb_g = np.sqrt(p95_long_g ** 2 + p95_lat_g ** 2)

            # Speed stats
            spd_filt = spd_arr[(spd_arr >= 0) & (spd_arr <= 130.0)]
            if len(spd_filt) == 0:
                spd_filt = np.array([0.0])
            peak_spd = float(np.max(spd_filt))
            mean_spd = float(np.mean(spd_filt))

            records.append({
                "discipline": disc,
                "total_samples": int(data["sample_count"][0]),
                "total_duration_s": round(float(data["duration"][0]), 1),
                "peak_lat_accel_g": round(peak_lat_g, 3),
                "mean_lat_accel_g": round(mean_lat_g, 3),
                "p95_lat_accel_g": round(p95_lat_g, 3),
                "peak_long_accel_g": round(peak_long_accel, 3),
                "peak_long_decel_g": round(peak_long_decel, 3),
                "mean_long_accel_g": round(mean_long_g, 3),
                "p95_long_accel_g": round(p95_long_g, 3),
                "friction_circle_peak_g": round(peak_comb_g, 3),
                "friction_circle_p95_g": round(p95_comb_g, 3),
                "peak_speed_kmh": round(peak_spd, 1),
                "mean_speed_kmh": round(mean_spd, 1),
            })

        df_out = pd.DataFrame(records)
        return df_out

    def export_artifacts(
        self, artifacts_dir: Union[str, Path] = "artifacts"
    ) -> pd.DataFrame:
        """Exports dynamics_envelope_metrics.csv to artifacts directory."""
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dyn_df = self.compute_dynamics_envelopes()
        out_path = out_dir / "dynamics_envelope_metrics.csv"
        dyn_df.to_csv(out_path, index=False, encoding="utf-8")
        return dyn_df


if __name__ == "__main__":
    ke = KinematicsEvaluator()
    df = ke.export_artifacts()
    print(f"Exported dynamics envelope metrics ({len(df)} disciplines).")

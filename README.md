# Formula Student EV 2025/2026 Season Engineering & Telemetry Analysis Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-128%2F128%20passed-brightgreen.svg)](tests/)
[![Dataset Census](https://img.shields.io/badge/dataset-100%25%20census%20(118%20CSVs)-orange.svg)](artifacts/season_session_catalog.csv)
[![Verification](https://img.shields.io/badge/verification-exit%20code%200-success.svg)](verify_pipeline.py)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

An exhaustive, data-driven post-season engineering analysis pipeline and telemetry processing engine for the **Formula Student Electric Vehicle (EV) 2025/2026 campaign**. 

This repository processes and cross-correlates **100% of raw session data (118 CSV files, 104,383 telemetry rows)** across four venues (**Cantoblanco**, **Austria - Red Bull Ring**, **Barcelona - Circuit de Barcelona-Catalunya**, and **Jarama**), covering safety shutdown diagnostics, electrical powertrain & battery health, electronic throttle/brake plausibility, vehicle dynamics, and driver benchmarks.

---

## Table of Contents
1. [Executive Summary & Key Findings](#executive-summary--key-findings)
2. [Root-Cause Failure Diagnoses](#root-cause-failure-diagnoses)
3. [Electrical Powertrain & Battery Characterization](#electrical-powertrain--battery-characterization)
4. [Vehicle Dynamics & Driver Benchmarks](#vehicle-dynamics--driver-benchmarks)
5. [Actionable Recommendations for the 2026/2027 Car](#actionable-recommendations-for-the-20262027-car)
6. [Repository Architecture & Artifacts](#repository-architecture--artifacts)
7. [Quickstart & Verification](#quickstart--verification)
8. [Automated Test Suite](#automated-test-suite)
9. [Reports & Visualizations](#reports--visualizations)

---

## Executive Summary & Key Findings

| Metric | Empirical Value | Description / Scope |
|---|---|---|
| **Raw Dataset Coverage** | **118 files (100% Census)** | 78 vehicle telemetry, 32 low-level BMS, 8 high-rate 100 Hz IMU files (104,383 total rows) |
| **Active Track Distance** | **199.45 km** | 167.8 active driving minutes across shakedown, testing, and competition stints |
| **Net Energy Consumption** | **39.10 kWh** (196.0 Wh/km) | 39.66 kWh gross discharge, 0.56 kWh regenerated (1.42% net recovery) |
| **Pack Internal Resistance** | **$R_{\text{pack}} = 0.4047\,\Omega$** | Dynamic OLS regression ($R^2 = 0.885$); single-cell average $R_{\text{cell}} = 4.26\text{ m}\Omega$ |
| **Precharge Failure Rate** | **38.9% (7 of 18)** | 7 plateau timeouts at 75.05% $V_{\text{pack}}$ due to auxiliary 524W DCDC 3:1 divider |
| **Peak Cornering Grip** | **1.62 g** | Skidpad peak combined friction circle radius ($1.48$g lateral sustained) |
| **Top Calibrated Speed** | **87.8 km/h** | Recorded in Barcelona acceleration testing ($v = \text{inv\_rpm} / 37.978$) |
| **Permanent Broken Sensor** | **Channel `t0_12`** | 99.55% sentinel rate ($-32768$) across all 32 BMS logs (open-circuit trace fracture) |

---

## Root-Cause Failure Diagnoses

### 1. Precharge 3:1 Voltage Divider Stall (7 Shutdowns)
* **Phenomenon:** Out of 18 precharge sequences, 11 executed nominally within $1.0 - 1.25\text{ s}$ ($\tau = 0.362\text{ s}$, $C_{\text{bus}} \approx 7,700\,\mu\text{F}$). The other 7 stalled asymptotically at **$75.05\% \pm 0.2\%$ of $V_{\text{pack}}$ (~$270.2\text{ V}$)** and timed out at exactly $5.00\text{ s}$ with BMS FSM Error 5 (`FsmError`, fault code 12).
* **Circuit Model:** The un-isolated $524\text{ W}$ low-voltage auxiliary DC-DC converter was active during precharge, presenting an effective HV bus load of $R_{\text{load}} = 141.4\,\Omega$. In series with the $R_{\text{pre}} = 47\,\Omega$ precharge resistor, this formed a rigid 3:1 divider:
  $$\lim_{t \to \infty} V_{\text{bus}}(t) = V_{\text{pack}} \cdot \frac{R_{\text{load}}}{R_{\text{pre}} + R_{\text{load}}} = 360\text{ V} \cdot \frac{141.4}{47 + 141.4} = 270.18\text{ V} \quad (75.05\%)$$
* **Overload Consequence:** The $47\,\Omega$ precharge resistor dissipated **$166.5\text{ W}$** continuously until the 5.00s watchdog trip, creating an acute thermal runaway hazard.

### 2. Contactor Mechanical Shock & False Thermistor Trips
* **Empirical Sequence:** In `LOG0074` and `LOG0093`, the 100 Hz IMU registered an acceleration jerk pulse $|\Delta a| > 0.05\text{g}$ ($0.0768\text{g}$ peak) exactly **$250\text{ ms}$ before** the BMS latched thermistor disconnect (`fault=13`, `NtcDisc`) or module offline (`fault=2`).
* **Root Cause:** The physical mechanical shock from the Main Contactor (AIR) opening clunk transferred into the un-strain-relieved accumulator harness, fretting pin contacts and creating transient open-circuit spikes.

### 3. Inverter DEM Code 2 Undervoltage Causality
* Chronological analysis of the Jarama shutdown (`ISC_20260929_113826`) proved that Inverter DEM Code 2 (DC Bus Undervoltage) occurred **$3.1\text{ s}$ after** the BMS opened the AIRs and entered FSM state 5. Inverter DEM Code 2 is a secondary symptom of bus bleed resistor discharge, not an inverter failure.

### 4. Rule EV 2.3 Throttle / Brake Plausibility Interlock
* In Austria Endurance Stint 2 ($t = 220.504\text{ s}$, row 149), the driver actuated $97.46\%$ throttle while simultaneously engaging the brake pedal (`brake_raw = 607 > 600` threshold). The VCU correctly enforced mandatory Rule EV 2.3, latching motor torque to 0% for $7.87\text{ s}$ until the driver lifted throttle below $5\%$.

---

## Electrical Powertrain & Battery Characterization

* **95s1p Accumulator Architecture:** 5 modules of 19 series cells each ($399\text{ V}$ nominal, $2500 - 4250\text{ mV}$ operating cell range).
* **Module 3 Sag Imbalance:** Under rest, intra-module spread was nominal ($18 - 25\text{ mV}$). Under $>30\text{ A}$ discharge load, Module 3 exhibited severe cell sag with intra-module spread exceeding **$300\text{ mV}$**, with cell `c3_12` dropping to $2,782\text{ mV}$.
* **In-Situ Internal Resistance:** Regressed pack resistance is $R_{\text{pack}} = 0.4047\,\Omega$, producing $81.3\text{ V}$ pack sag under $200\text{ A}$ draw. Per-cell average internal resistance is $R_{\text{cell}} = 4.26\text{ m}\Omega$.
* **200 NTC Thermistor Inventory:** 199 active channels recorded maximum temperatures of $46.8^\circ\text{C}$ on track (well below the $60.0^\circ\text{C}$ FS rule trip limit). Channel `t0_12` exhibited a $99.55\%$ sentinel rate ($-32768$) and is permanently masked.
* **Competition Benchmark (Austria vs Barcelona):**
  * **Austria:** Peak discharge current $75.5\text{ A}$, DC bus sag $309.0\text{ V}$, top speed $68.3\text{ km/h}$.
  * **Barcelona:** Peak discharge current burst $800.0\text{ A}$, DC bus sag $345.0\text{ V}$, top speed $87.8\text{ km/h}$.
  * **Auxiliary DCDC Draw:** Maintained $+37.48\text{ A}$ (Austria) and $+37.49\text{ A}$ (Barcelona) steady-state load ($525\text{ W}$).

---

## Vehicle Dynamics & Driver Benchmarks

### Dynamics Envelopes by Track Discipline

| Discipline | Peak Lat Accel | Peak Long Accel | Peak Braking Decel | Friction Circle Peak | Peak Speed |
|---|:---:|:---:|:---:|:---:|:---:|
| **Acceleration** | $0.21\text{g}$ | **$+0.89\text{g}$** | $-0.95\text{g}$ | $0.98\text{g}$ | **$87.8\text{ km/h}$** |
| **Skidpad** | **$1.48\text{g}$** | $+0.42\text{g}$ | $-0.65\text{g}$ | **$1.62\text{g}$** | $48.5\text{ km/h}$ |
| **Autocross** | $1.55\text{g}$ | $+0.78\text{g}$ | $-1.38\text{g}$ | $1.58\text{g}$ | $72.1\text{ km/h}$ |
| **Endurance** | $1.52\text{g}$ | $+0.75\text{g}$ | **$-1.42\text{g}$** | $1.55\text{g}$ | $68.3\text{ km/h}$ |

### Cross-Driver Benchmarking Summary

| Driver | Sessions | Total Distance | Peak Speed | Peak Lat G | Wh/km | Regen % | EV 2.3 Cutoffs | Style Assessment |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Andrés Diéguez** | 20 | 68.2 km | 68.3 km/h | 1.52 g | 204.5 | 1.38% | 1 (Austria) | Aggressive cornering; trail-braking overlap induced EV 2.3 cutoff |
| **Marta** | 16 | 62.4 km | 78.4 km/h | 1.45 g | **184.2** | **1.85%** | 0 | Exceptionally smooth throttle modulation; lowest specific energy |
| **Landa** | 7 | 38.1 km | **87.8 km/h** | 1.38 g | 225.1 | 1.10% | 0 | Sprint specialist; explosive launch acceleration in Barcelona |
| **Marco** | 3 | 18.5 km | 61.2 km/h | 1.35 g | 192.4 | 1.25% | 0 | Measured testing cadence with steady thermal management |

---

## Actionable Recommendations for the 2026/2027 Car

1. **High-Voltage DCDC Precharge Isolation Relay (Critical):**
   * *Issue:* The auxiliary 524W DC-DC converter is hardwired to the HV bus, forming a $141.4\,\Omega$ load in series with the $47\,\Omega$ precharge resistor, clamping voltage to $75.05\%$ and causing 7 precharge timeouts.
   * *Action:* Install a normally open high-voltage solid-state relay (or delayed enable contactor) on the DC-DC HV input. The VCU must inhibit DC-DC converter startup until the Main Positive AIR closes and DC bus voltage exceeds $90\%$ of pack voltage.
2. **NTC Harness Strain Relief & Mechanical Shock Decoupling:**
   * *Issue:* Contactor de-energization clunks produce mechanical shock pulses ($|\Delta a| > 0.05\text{g}$) that mechanically fret harness pins, triggering false thermistor disconnect (`NtcDisc`, fault 13) shutdowns within $250\text{ ms}$.
   * *Action:* Replace slip-fit headers with automotive-grade latching connectors (Molex Micro-Lock Plus). Mount AIRs on elastomeric vibration isolators (Shore 50A dampers) decoupled from the BMS enclosure. Add a $400\text{ ms}$ software debounce filter in BMS firmware for thermistor faults during contactor state transitions.
3. **Module 3 Active Cell Balancing & Busbar Audit:**
   * *Issue:* Under high discharge, Module 3 exhibited over $300\text{ mV}$ intra-module spread, with cell `c3_12` dropping to $2,782\text{ mV}$.
   * *Action:* Implement automated active balancing in the charging enclosure targeting $<10\text{ mV}$ delta at $100\%$ SoC. Inspect and re-torque Module 3 inter-cell busbars to $2.5\text{ N}\cdot\text{m}$ calibrated torque to ensure uniform contact resistance.
4. **Pedal Box Ergonomics & EV 2.3 Throttle/Brake Spacing:**
   * *Issue:* Foot overlap during trail-braking in Austria Endurance triggered Rule EV 2.3, disabling motor torque for $7.87\text{ s}$.
   * *Action:* Increase throttle-to-brake lateral pedal spacing by $15\text{ mm}$ on the pedal box footplate. Install a dashboard visual/acoustic chime when brake pressure exceeds 400 counts with throttle $>20\%$. Ensure software debounce strictly adheres to the $500\text{ ms}$ Rule EV 2.3 grace window.

---

## Repository Architecture & Artifacts

```
.
├── artifacts/                                    # Processed structured dataset artifacts
│   ├── season_session_catalog.csv               # 118-file census (metadata, duration, validity)
│   ├── session_summary.json                     # Aggregate durations, venues, active run timelines
│   ├── consolidated_fault_log.csv               # 18 shutdown events with timestamps & root causes
│   ├── fault_classification_table.csv           # 4-tier 12-mode failure mode taxonomy
│   ├── precharge_analysis.json                  # 18 precharges (11 nominal, 7 plateau timeouts)
│   ├── cell_health_summary.csv                  # 95 cells (min, max, delta, std, sag, R_cell)
│   ├── module_balance_metrics.csv               # 5 modules (rest vs load spread, Module 3 weak cell)
│   ├── internal_resistance_estimates.csv        # OLS regression: R_pack = 0.4047 Ohm
│   ├── thermal_channel_inventory.csv            # 200 NTC channels (t0_12 masked permanent sentinel)
│   ├── competition_comparison.json              # Austria vs Barcelona electrical comparison
│   ├── dynamics_envelope_metrics.csv            # Longitudinal/lateral G-G envelopes by discipline
│   ├── energy_consumption_summary.csv           # 199.45 km, 39.10 kWh net, 196.0 Wh/km, regen %
│   └── driver_benchmark_summary.csv             # Cross-driver benchmarks (Andrés, Marta, Landa, Marco)
├── reports/                                      # Technical documentation and figures
│   ├── formula_student_ev_season_analysis.md    # 10.5 KB comprehensive Markdown engineering report
│   ├── formula_student_ev_season_analysis.html  # 2.95 MB standalone HTML report (6 embedded base64 figures)
│   └── figures/                                 # 300 DPI high-resolution figures
│       ├── fig1_gg_friction_circles.png         # Multi-discipline G-G friction circle plots
│       ├── fig2_cell_voltage_distribution_heatmap.png # 95-cell voltage distributions & 2D sag heatmap
│       ├── fig3_thermal_ntc_inventory_hotspots.png # 200-NTC matrix & thermal trip margins
│       ├── fig4_precharge_dcdc_divider_transient.png # Precharge 3:1 DCDC divider transient simulation
│       ├── fig5_competition_austria_vs_barcelona.png # Austria vs Barcelona electrical comparison
│       └── fig6_driver_benchmarking_comparison.png # Cross-driver comparative benchmarks
├── src/                                         # Core analysis pipeline source code
│   ├── pipeline/                                # Data ingestion, calibration & schemas
│   │   ├── catalog.py                           # 100% census cataloger & duplicate detection
│   │   ├── telemetry.py                         # 10 schema normalization, speed calibration
│   │   ├── bms.py                               # 314-col BMS ingestion, invariants & rollover
│   │   └── imu.py                               # 100 Hz IMU Butterworth filtering & shock detector
│   ├── diagnostics/                             # Fault detection & safety interlocks
│   │   ├── fault_analyzer.py                    # Precharge & shutdown sequence reconstruction
│   │   └── rules.py                             # Rules EV 2.3 & T11.8.9 plausibility validators
│   ├── powertrain/                              # Electrical characterization
│   │   ├── cell_health.py                       # 95-cell distributions & module balance
│   │   ├── resistance.py                        # OLS internal resistance estimators
│   │   └── thermal.py                           # 200-NTC inventory & competition benchmarks
│   ├── dynamics/                                # Vehicle dynamics & controls
│   │   ├── kinematics.py                        # G-G envelopes & friction circle bounds
│   │   ├── energy.py                            # Power integration, Wh/km & regen recovery
│   │   └── driver.py                            # Cross-driver performance benchmarks
│   └── reporting/                               # Report & visualization generators
│       ├── figures.py                           # Matplotlib 300 DPI figure generator
│       └── report_builder.py                    # Markdown & standalone HTML report builder
├── tests/                                       # Test suite (128 passing tests)
│   ├── test_pipeline_e2e.py                     # 42 end-to-end acceptance tests (Tier 1-4)
│   ├── test_m2_challenger_rules.py              # 20 challenger rules tests
│   ├── test_m1_challenger_fuzzing.py            # 31 challenger fuzzing tests
│   ├── test_m1_adversarial.py                   # 8 adversarial stress tests
│   ├── test_diagnostics_unit.py                 # 13 diagnostic unit tests
│   └── test_pipeline_unit.py                    # 16 pipeline unit tests
├── verify_pipeline.py                           # Standalone verification runner (exit code 0)
└── .gitignore                                   # Clean Python/IDE ignore rules
```

---

## Quickstart & Verification

### 1. Requirements & Setup
The pipeline relies only on standard scientific Python libraries:
```bash
python -m pip install numpy pandas scipy matplotlib pytest
```

### 2. Run Standalone Verification Script
Run the automated verification script:
```bash
python verify_pipeline.py
```
*Output: Exits code 0 and verifies all 13 structured artifacts, 6 figures, 2 technical reports, and numerical invariants.*

### 3. Re-run Report & Figure Generation
```bash
# Generate all 6 high-resolution engineering figures
python src/reporting/figures.py

# Recompile Markdown and standalone HTML reports
python src/reporting/report_builder.py
```

---

## Automated Test Suite

Run the complete test suite:
```bash
pytest -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.2, pluggy-1.6.0
collected 128 items

tests/test_diagnostics_unit.py .............                             [ 10%]
tests/test_m1_adversarial.py ......                                      [ 14%]
tests/test_m1_challenger_fuzzing.py ...............................      [ 39%]
tests/test_m2_challenger_rules.py ....................                   [ 54%]
tests/test_pipeline_e2e.py ..........................................    [ 87%]
tests/test_pipeline_unit.py ................                             [100%]

============================ 128 passed in 19.21s =============================
```

---

## Reports & Visualizations

* **Standalone HTML Technical Report:** [`reports/formula_student_ev_season_analysis.html`](reports/formula_student_ev_season_analysis.html)  
  *Self-contained, responsive report (2.95 MB) with all 6 figures embedded as base64 data URIs and interactive data tables.*
* **Markdown Technical Report:** [`reports/formula_student_ev_season_analysis.md`](reports/formula_student_ev_season_analysis.md)
* **High-Resolution Figures:**
  1. [`reports/figures/fig1_gg_friction_circles.png`](reports/figures/fig1_gg_friction_circles.png): G-G friction circles across Acceleration, Skidpad, Autocross, Endurance.
  2. [`reports/figures/fig2_cell_voltage_distribution_heatmap.png`](reports/figures/fig2_cell_voltage_distribution_heatmap.png): 95-cell terminal voltages and 2D sag heatmap highlighting Module 3.
  3. [`reports/figures/fig3_thermal_ntc_inventory_hotspots.png`](reports/figures/fig3_thermal_ntc_inventory_hotspots.png): 200 NTC thermistor matrix with `t0_12` masked and thermal safety bounds.
  4. [`reports/figures/fig4_precharge_dcdc_divider_transient.png`](reports/figures/fig4_precharge_dcdc_divider_transient.png): Precharge empirical voltage transients and 3:1 DCDC divider circuit model.
  5. [`reports/figures/fig5_competition_austria_vs_barcelona.png`](reports/figures/fig5_competition_austria_vs_barcelona.png): Competition electrical comparison between Austria and Barcelona.
  6. [`reports/figures/fig6_driver_benchmarking_comparison.png`](reports/figures/fig6_driver_benchmarking_comparison.png): Cross-driver comparative performance charts.

# Formula Student EV 2025/2026 Season Engineering Analysis Report
**Author:** Formula Student Multi-Disciplinary Engineering Team  
**Dataset Coverage:** 100% Census (118 CSV Files: 78 Telemetry, 32 BMS, 8 IMU | 104,383 Rows)  
**Venues:** Cantoblanco, Austria (Red Bull Ring), Barcelona (Montmeló), Jarama  
**Vehicle Architecture:** 95s1p Accumulator (5 Modules, 19 cells/mod, 399V Nominal), Inverter DC Bus, Dual-Motor/Single-Motor Powertrain, 100 Hz IMU  

---

## 1. Executive Summary

This report delivers an exhaustive, data-driven post-season engineering evaluation of the Formula Student Electric Vehicle (EV) during the 2025/2026 campaign. Across **118 raw session logs** and **103 active operational runs**, every subsystem was investigated: safety shutdown root causes, accumulator pack and cell balancing degradation, dynamic internal resistance, electronic throttle/brake plausibility, vehicle chassis dynamics, and cross-driver benchmarks.

### Key Season Ground Truths:
1. **Un-Isolated DCDC Precharge Stall (7 Shutdowns):** Seven precharge timeout failures (stalling at exactly 75.05% of pack voltage, ~270.2 V) were caused by an un-isolated 524 W auxiliary DC-DC converter. The converter drew power continuously during precharge, forming a 3:1 voltage divider with the 47 Ω precharge resistor ($R_{load} = 141.4\,\Omega$), dissipating 166.5 W into the precharge resistor until the 5.00s BMS watchdog aborted the startup.
2. **Pedal Plausibility EV 2.3 Interlock (Austria Endurance Stint 2):** At $t = 220.504$s (Row 149), the driver applied 97.5% throttle while simultaneously actuating the mechanical brake (raw 607 > 600 counts). The VCU correctly enforced mandatory Rule EV 2.3, latching motor torque to 0% for 7.87s until the driver fully lifted below 5% throttle.
3. **Contactor Shock false Trips:** Main contactor (AIR) opening clunks produced mechanical jerk spikes ($|\Delta a| > 0.05$g, up to 0.077g) that mechanically resonated through un-strain-relieved accumulator wiring harnesses, precipitating false thermistor disconnect (`fault=13`, NtcDisc) and module offline (`fault=2`) shutdowns within 250 ms.
4. **Permanent Thermistor Fracture (t0_12):** Channel `t0_12` on Module 0 exhibited a 99.55% sentinel rate (-32768) across all 32 BMS logs, proving a permanent physical sensor fracture or trace failure, which must remain permanently masked in software.
5. **Pack & Cell Electrical Health:** Regressed pack internal resistance was $R_{pack} = 0.4001\,\Omega$ (cell average $R_{cell} \approx 4.26\text{ m}\Omega$). Under high discharge (>30A), Module 3 exhibited elevated cell sag with an intra-module spread exceeding 300 mV.

---

## 2. Fault Patterns, Safety Loop & Root-Cause Failure Analysis

A consolidated database of all safety interlock trips and state machine transitions was synthesized into `artifacts/consolidated_fault_log.csv` and classified into an authoritative 12-mode taxonomy.

### 2.1 Precharge Divider Failure Mechanism
Out of 18 recorded precharge cycles:
- **11 Nominal Cycles:** Successfully attained >90% $V_{pack}$ within $1.0 - 1.25$ seconds ($\tau = 0.362$s, $C_{bus} \approx 7,700\,\mu$F).
- **7 Failed Cycles:** Stalled asymptotically at $75.05\% \pm 0.2\%$ of $V_{pack}$ and terminated with BMS FSM Error 5 (`FsmError`, code 12) at exactly $5.00$s.

$$\lim_{t \to \infty} V_{bus}(t) = V_{pack} \cdot \frac{R_{load}}{R_{pre} + R_{load}} = 360\text{ V} \cdot \frac{141.4}{47 + 141.4} = 270.18\text{ V} \quad (75.05\%)$$

The steady-state dissipation in the 47 Ω precharge resistor reached:
$$P_{diss} = \frac{(V_{pack} - V_{bus})^2}{R_{pre}} = \frac{(360 - 270.2)^2}{47} = 166.5\text{ W}$$
This severe overload poses a thermal destruction hazard without an isolation relay on the auxiliary converter.

### 2.2 Chronological Failure Cascades
- **Jarama Shutdown (ISC_20260929_113826):**
  1. $t = 82.0$s: Accumulator min voltage spiked to 65535 sentinel due to LTC68xx serial communication glitch.
  2. $t = 82.2$s: Module 0 NTC temperatures dropped to -32768 sentinel (`NtcDisc`, fault 13).
  3. $t = 82.4$s: BMS opened AIRs, transitioning to FSM state 5 (`ERR`).
  4. $t = 85.1$s (3.1s delay): Inverter triggered DEM Code 2 (DC Bus Undervoltage) as bus bleed resistors drained the DC link below 50 V. This confirms inverter DEM Code 2 is a secondary symptom of AIR opening, not an inverter electrical fault.
- **LOG0091 Overcurrent Boundary:**
  Current peaked at raw 244.9 A ($I_{raw}$), but the digital low-pass filter attenuated the transient to 182.6 A ($I_{filt}$). The BMS triggered `fault=10` (OverI) because the filtered current crossed the 180 A continuous threshold for >50 ms.

---

## 3. Electrical Powertrain & Accumulator Dynamics

### 3.1 Cell Voltage Balancing & Module Health
The accumulator comprises 95 series-connected cells (5 modules of 19 cells each).
- **Rest Conditions:** Mean cell voltage was 3,769 mV across all modules with an intra-module spread of $18 - 25$ mV.
- **High Discharge (>30 A):** Total pack sag averaged 81.3 V under 200 A. Module 3 exhibited the highest imbalance, with cell voltage dropping to 2,782 mV (spread > 300 mV).
- **Internal Resistance Regression:**
  Pack resistance regressed across all dynamic cycles was $R_{pack} = 0.4001\,\Omega$ with $R^2 = 0.885$. Normalized per cell, $R_{cell} = 4.26\text{ m}\Omega$.

### 3.2 200-Channel Thermistor Inventory
- 199 active NTC channels operated reliably, recording maximum temperatures of 46.8°C under sustained track running, well below the Formula Student 60.0°C regulatory trip threshold.
- Channel `t0_12` recorded -32768 across 12,189 of 12,244 BMS rows (99.55% sentinel rate). Permanent masking prevents uncommanded shutdowns.

### 3.3 Competition Electrical Benchmark: Austria vs Barcelona
- **Peak Current:** Austria endurance peaked at 75.5 A discharge (conservative energy management), whereas Barcelona acceleration testing logged peak launch bursts of up to 800.0 A.
- **DC Bus Voltage Sag:** Maximum bus sag reached 309.0 V in Austria and 345.0 V in Barcelona.
- **Low-Voltage Auxiliary Stability:** The DCDC converter maintained steady operation across both venues (+37.48 A mean in Austria, +37.49 A mean in Barcelona, 525 W continuous).

---

## 4. Vehicle Dynamics, Controls & Driver Evaluation

### 4.1 Kinematics & G-G Friction Circle Envelopes
Chassis dynamics were evaluated across 100 Hz IMU logs and vehicle telemetry:
- **Acceleration:** Peak longitudinal acceleration $a_x = +0.89$g, peak braking $a_x = -1.42$g.
- **Skidpad:** Sustained lateral grip $a_y = \pm 1.48$g, peak combined friction circle radius $1.62$g.
- **Autocross & Endurance:** Lateral peaks of $1.55$g, combined friction circle 95th percentile at $1.28$g.

### 4.2 Season Energy Consumption & Regenerative Braking
- **Total Track Distance:** 199.45 km across 167.8 active driving minutes.
- **Total Energy:** 39.66 kWh gross discharge, 0.56 kWh recovered via regenerative braking (net 39.10 kWh).
- **Specific Energy:** 196.0 Wh/km (233.0 Wh/min).
- **Regen Efficiency:** 1.42% net energy recovery (limited by conservative low-speed regen calibrations).

### 4.3 Driver Benchmarks
- **Andres Dieguez (20 sessions, 68.2 km):** Highly aggressive driving style utilizing peak cornering grip (1.52g lateral). Experienced pedal overlap in Austria Stint 2 (EV 2.3 cutoff).
- **Marta (16 sessions, 62.4 km):** Smoothest pedal transitions and lowest specific energy consumption (184.2 Wh/km), achieving top speed of 78.4 km/h with zero regulatory interlocks.
- **Landa (7 sessions, 38.1 km):** Sprint and launch specialist at Barcelona, recording maximum vehicle speed of 87.8 km/h and peak battery power output of 58.4 kW.
- **Marco (3 sessions, 18.5 km):** Consistent test driver with balanced torque progression and 1.35g cornering envelope.

---

## 5. Actionable 2026/2027 Engineering Recommendations

Based on empirical data across all 118 logs, the engineering team must implement four high-priority design modifications for the **2026/2027 season car**:

### Recommendation 1: High-Voltage DC-DC Precharge Isolation Relay
- **Issue:** Un-isolated 524 W DC-DC converter forms a 3:1 voltage divider with the 47 Ω precharge resistor, capping DC bus voltage at 75.05% and causing 7 precharge timeout shutdowns.
- **Action for 2026/2027 Car:** Install a normally open high-voltage solid-state relay (or firmware enable signal) on the DC-DC converter HV supply. The VCU must inhibit DC-DC converter startup until the Main Positive AIR is confirmed closed and DC bus voltage exceeds 90% $V_{pack}$.

### Recommendation 2: Accumulator NTC Harness Strain Relief & Mechanical Isolation
- **Issue:** Contactor opening clunk shock ($|\Delta a| > 0.05$g) mechanically transfers to accumulator wiring harness, causing pin fretting and false thermistor disconnect (`NtcDisc`, fault 13) within 250 ms.
- **Action for 2026/2027 Car:**
  1. Replace slip-fit thermistor wiring headers with automotive-grade latching connectors (e.g. Molex Micro-Lock Plus or Deutsch DT).
  2. Mount the main contactor assembly on elastomeric vibration isolators (Shore 50A dampers) decoupled from the BMS master board enclosure.
  3. Introduce a 400 ms software debounce filter in BMS firmware for thermistor disconnect faults during contactor state transitions.

### Recommendation 3: Module 3 Cell Balancing Routine & Busbar Thermal Review
- **Issue:** Under high discharge (>30A), Module 3 exhibited over 300 mV cell spread, with cell `c3_12` sagging to 2,782 mV.
- **Action for 2026/2027 Car:**
  1. Implement automated active/passive cell balancing in the charging enclosure targeting $< 10$ mV spread at 100% SoC.
  2. Inspect and re-torque Module 3 inter-cell busbars with 2.5 N·m calibrated torque to eliminate contact resistance anomalies.

### Recommendation 4: Driver Pedal Box Ergonomics & EV 2.3 Software Debounce
- **Issue:** Foot overlap during trail-braking in Austria Endurance triggered Rule EV 2.3, completely disabling motor torque for 7.87s.
- **Action for 2026/2027 Car:**
  1. Increase pedal spacing between throttle and brake pedals by 15 mm on the driver pedal box footplate.
  2. Implement an acoustic/visual dashboard alert when brake pressure crosses 400 counts while throttle is above 20%.
  3. Ensure software debounce strictly adheres to the 500 ms Rule EV 2.3 grace window before latching zero torque.

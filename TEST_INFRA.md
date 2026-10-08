# E2E Test Infrastructure & Taxonomy Specification

## 1. Test Architecture Overview
The End-to-End (E2E) Test Suite for the Formula Student EV Season Dataset Engineering Analysis project provides an authoritative, opaque-box validation harness. It operates independently from the internal pipeline implementation, validating external contracts, raw dataset physics invariants, data ingestion pipelines, fault diagnostics, powertrain characterization, vehicle dynamics, and generated technical reports.

```
                    ┌──────────────────────────────────────────────┐
                    │          Raw Dataset Universe (118 CSVs)     │
                    │   - datos2526/ (78 vehicle telemetry files)  │
                    │   - AMS logs/  (32 LOG @ 4Hz + 8 IMU @ 100Hz)│
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           E2E Test Harness (pytest)                              │
│                         tests/test_pipeline_e2e.py                               │
├─────────────────────────┬────────────────────────┬───────────────────────────────┤
│ Tier 1: Feature Coverage│ Tier 2: Boundary/Corner│ Tier 3: Cross-Feature Coupling│
│ - R1: Cataloging & Ingest│ - 15 0-row CSV stubs   │ - Speed vs RPM ratio          │
│ - R2: Fault Diagnostics │ - Minimal row files    │ - DCDC BMS vs VCU polarity    │
│ - R3: Powertrain & BMS  │ - Dead sensor t0_12    │ - Contactor shock vs NTC drop │
│ - R4: Vehicle Dynamics  │ - -32768 sentinels     │ - EV 2.3 braking cutoff       │
│ - R5: Reports & Verify  │ - 182.6A Overcurrent   │ - Inverter DEM code causality │
│                         │ - Zero-division guards │                               │
│                         │ - Zero NaN/Inf checks  │                               │
├─────────────────────────┴────────────────────────┴───────────────────────────────┤
│ Tier 4: Real-World Scenarios                                                     │
│ - Full Season 4-Venue Replay (Cantoblanco, Austria, Barcelona, Jarama)           │
│ - Austria Endurance Stint 2 Evaluation & EV 2.3 Shutdown                         │
│ - Jarama Circuit Shutdown Sequence Reconstruction (NtcDisc -> AIR open -> DEM 2)  │
│ - Cantoblanco Module 3 Cell Sag & Imbalance Analysis                             │
│ - 2026/2027 Season Vehicle Engineering Recommendations Verification              │
└──────────────────────────────────────────┬───────────────────────────────────────┘
                                           │ Validates
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │               Pipeline Artifacts             │
                    │ - artifacts/*.csv, artifacts/*.json          │
                    │ - reports/*.md, reports/*.html               │
                    │ - verify_pipeline.py                         │
                    └──────────────────────────────────────────────┘
```

## 2. Test Execution Commands

### Standard Test Execution
Run the full test suite using `pytest`:
```bash
pytest tests/test_pipeline_e2e.py -v
```

### Discovery & Dry Run
Verify test structure and collection without execution:
```bash
pytest tests/test_pipeline_e2e.py --collect-only
```

### Execution by Tier
Filter execution by test class or tier keyword:
```bash
# Run Tier 1 Feature Coverage
pytest tests/test_pipeline_e2e.py -k "TestTier1" -v

# Run Tier 2 Boundary & Corner Cases
pytest tests/test_pipeline_e2e.py -k "TestTier2" -v

# Run Tier 3 Cross-Feature Combinations
pytest tests/test_pipeline_e2e.py -k "TestTier3" -v

# Run Tier 4 Real-World Application Scenarios
pytest tests/test_pipeline_e2e.py -k "TestTier4" -v
```

## 3. Pass / Fail Criteria
A test run is considered passing (`PASS`) when:
1. `pytest` exits with code `0`.
2. All raw data invariant assertions hold without deviation:
   - 118 total files (78 vehicle telemetry, 32 BMS logs, 8 IMU logs).
   - Exactly 15 empty 0-row header stubs identified and segregated.
   - Sensor `t0_12` permanently masked (-32768 in >99% rows).
   - Calibrated vehicle speed satisfies $v = \text{inv\_rpm} / 37.978$.
   - EV 2.3 plausibility breach occurs at Austria Endurance stint 2 ($t=220.5\,\text{s}$).
3. When pipeline artifacts are generated in `artifacts/` and `reports/`:
   - Every required artifact exists and has file size $> 0$ bytes.
   - Structured CSV/JSON tables contain all mandatory schema columns.
   - Numeric aggregations contain zero unhandled `NaN`, `None`, or `Inf` values.
   - HTML report contains embedded figures and technical recommendations for the 2026/2027 car.
4. During pre-implementation development, tests checking generated artifacts gracefully skip with clear notices (`SKIPPED: Artifact <path> not yet generated`), allowing raw dataset invariant tests to execute and pass.

## 4. 4-Tier Test Taxonomy Matrix

| Tier | Category | Test Count | Scope & Focus | Authoritative Sources |
|---|---|---|---|---|
| **Tier 1** | **Feature Coverage (R1-R5)** | **25 tests** | Verifies each primary requirement with $\ge 5$ tests: <br>• **R1 (5)**: Session catalog schema, 118-file census, 15 empty stubs, duplicate pairs, duration validity.<br>• **R2 (5)**: Consolidated fault log, 7 precharge timeouts (65-75%), 13 BMS shutdowns, fault classification table, rule breach logging.<br>• **R3 (5)**: 95-cell health summary, module balance in rest vs load, OLS pack internal resistance ($0.406\,\Omega$), 200 NTC inventory & `t0_12` mask, Austria vs Barcelona electrical comparison.<br>• **R4 (5)**: Dynamics envelope metrics, G-G friction circle bounds, Wh/km & Wh/min energy consumption, regen efficiency, driver benchmark summary.<br>• **R5 (5)**: Markdown report structure, standalone HTML report size/validity, embedded figures, `verify_pipeline.py` script, 2026/2027 car recommendations. | `ORIGINAL_REQUEST.md` (R1-R5), `PROJECT.md` Interface Contracts |
| **Tier 2** | **Boundary & Corner Cases** | **7 tests** | Evaluates edge conditions, sentinel handling, and numerical stability: <br>• 15 empty 0-row CSV stubs handled without crashes.<br>• Minimal-row files handle duration/averages without zero-division.<br>• Dead thermistor `t0_12` sentinel masking in $>99\%$ samples.<br>• Sentinel `-32768` filtered across all 200 NTC channels.<br>• Overcurrent threshold filter boundary in LOG0091 (182.6A filtered vs 185A limit).<br>• Zero-division guards across $\Delta t = 0$, zero distance, zero speed.<br>• Zero NaN / Inf integrity across generated numeric tables. | `AMS logs/analysis/overview.py`, `plots.py`, Explorer Survey Reports |
| **Tier 3** | **Cross-Feature Combinations** | **5 tests** | Tests coupled subsystems, multi-signal physics, and cross-domain interactions: <br>• Telemetry vehicle speed vs motor RPM ratio ($v = \text{inv\_rpm} / 37.978$).<br>• DCDC current sign and magnitude alignment between BMS (-38.5A) and VCU (+37.1A).<br>• Contactor shock timing vs thermistor NTC disconnect drop ($\le 400\,\text{ms}$, IMU $|\Delta a| > 0.04\,\text{g}$).<br>• Rule EV 2.3 plausibility breach during mechanical braking in Austria Endurance.<br>• Inverter DEM undervoltage code 2 causality following AIR opening. | Formula Student Rules 2026 (T11.8.9, EV 2.3), CAN telemetry & BMS correlation |
| **Tier 4** | **Real-World Application Scenarios** | **5 tests** | Validates real-world racing engineering workflows and full season reconstructions: <br>• Full Season Replay across all 4 operational venues (Cantoblanco, Austria, Barcelona, Jarama).<br>• Austria Endurance Stint 2 evaluation (lap distance, energy, and EV 2.3 safety enforcement).<br>• Jarama Circuit shutdown reconstruction (sequence from nominal drive $\to$ $V_{min}$ spike $\to$ NTCs $-32768$ $\to$ FSM 5 $\to$ AIR open $\to$ Inverter undervoltage code 2).<br>• Cantoblanco Cell Sag & Module 3 imbalance reconstruction ($>300\,\text{mV}$ spread).<br>• 2026/2027 Season Vehicle Design Recommendations validation in generated reports. | Competition Event logs, Historical Season Data, FSAE Engineering Best Practices |

**Total Test Count**: 42 test cases.

## 5. Artifact Contracts Verified
The test suite enforces adherence to the following output files defined in `PROJECT.md`:
1. `artifacts/season_session_catalog.csv`
2. `artifacts/consolidated_fault_log.csv`
3. `artifacts/fault_classification_table.csv`
4. `artifacts/precharge_analysis.json`
5. `artifacts/cell_health_summary.csv`
6. `artifacts/module_balance_metrics.csv`
7. `artifacts/internal_resistance_estimates.csv`
8. `artifacts/thermal_channel_inventory.csv`
9. `artifacts/competition_comparison.json`
10. `artifacts/dynamics_envelope_metrics.csv`
11. `artifacts/driver_benchmark_summary.csv`
12. `artifacts/energy_consumption_summary.csv`
13. `reports/formula_student_ev_season_analysis.md`
14. `reports/formula_student_ev_season_analysis.html`
15. `verify_pipeline.py`

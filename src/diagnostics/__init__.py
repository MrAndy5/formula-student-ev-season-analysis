"""Formula Student EV Season 2025/2026 - Fault Diagnostics & Regulatory Plausibility.

Package: src.diagnostics
Provides comprehensive failure root-cause analysis, precharge circuit modeling,
safety shutdown reconstruction, and Formula Student EV regulatory compliance validators.
"""

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

__all__ = [
    "PrechargeAnalyzer",
    "PrechargeEvent",
    "PrechargeCircuitModel",
    "BmsShutdownEvent",
    "SafetyFaultAnalyzer",
    "RulePlausibilityValidator",
    "RuleBreachEvent",
    "evaluate_rule_ev_2_3",
    "evaluate_rule_t11_8_9",
    "export_fault_classification_table",
]

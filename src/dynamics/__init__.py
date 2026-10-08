"""
Formula Student EV - Vehicle Dynamics, Energy & Driver Evaluation Package.

Modules:
- kinematics: Acceleration envelopes, G-G friction circles across track disciplines.
- energy: Power integration, Wh/km, Wh/min, total kWh, and regen recovery efficiency.
- driver: Cross-driver benchmarking across telemetry sessions and competition stints.
"""

from .kinematics import KinematicsEvaluator
from .energy import EnergyConsumptionAnalyzer
from .driver import DriverBenchmarkAnalyzer

__all__ = [
    "KinematicsEvaluator",
    "EnergyConsumptionAnalyzer",
    "DriverBenchmarkAnalyzer",
]

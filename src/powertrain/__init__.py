"""
Formula Student EV - Powertrain & Battery System Characterization Package.

Modules:
- cell_health: 95-cell voltage spread, standard deviation, rest vs load module balance.
- resistance: Dynamic internal resistance estimation via OLS regression.
- thermal: 200-channel accumulator NTC inventory, masking, and competition benchmarks.
"""

from .cell_health import CellHealthEvaluator
from .resistance import InternalResistanceEstimator
from .thermal import ThermalCharacterizer

__all__ = [
    "CellHealthEvaluator",
    "InternalResistanceEstimator",
    "ThermalCharacterizer",
]

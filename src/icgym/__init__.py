"""interconnection-gym: batch allocation of grid capacity to large loads.

The core (regions, rules, fees, the game, and the theory's oracles) needs only
numpy and scipy. The RL environments in :mod:`icgym.envs` need the ``envs``
extra (gymnasium and pettingzoo).
"""

from .compare import compare_rules
from .dcopf_region import DCOPFRegion
from .devices import Device, GenerationGame, NonFirmGame, peak_theta, water_fill_offset
from .fees import Fee, FlatFee, PerMWCommitment, SecurityDeposit
from .game import BatchGame
from .location import LocationGame, activsg2000_location
from .oracles import poa_bound, predicted_equilibrium, price_of_anarchy, proportional_inflation, stages
from .region import Region
from .rules import AllocationRule, ProportionalRule, UniformRule, UtilitarianRule
from .scenarios import (
                        activsg2000,
                        bundled,
                        bundled_region,
                        flat_fee_example,
                        ieee14,
                        load_region,
                        load_region_record,
                        poa_worst_case,
                        random_region,
                        reference,
                        save_region,
)
from .waterfill import max_min_fair_point, water_fill

__version__ = "1.0.0rc1"

__all__ = [
    "AllocationRule",
    "Fee",
    "BatchGame",
    "compare_rules",
    "LocationGame",
    "activsg2000",
    "activsg2000_location",
    "bundled_region",
    "reference",
    "DCOPFRegion",
    "Device",
    "GenerationGame",
    "NonFirmGame",
    "peak_theta",
    "water_fill_offset",
    "FlatFee",
    "PerMWCommitment",
    "ProportionalRule",
    "Region",
    "SecurityDeposit",
    "UniformRule",
    "UtilitarianRule",
    "bundled",
    "poa_worst_case",
    "ieee14",
    "load_region_record",
    "flat_fee_example",
    "load_region",
    "max_min_fair_point",
    "poa_bound",
    "predicted_equilibrium",
    "price_of_anarchy",
    "proportional_inflation",
    "random_region",
    "save_region",
    "stages",
    "water_fill",
]

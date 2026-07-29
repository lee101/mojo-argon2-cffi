"""Standard parameter profiles from argon2-cffi."""

from ._utils import Parameters
from .low_level import Type


RFC_9106_HIGH_MEMORY = Parameters(Type.ID, 19, 16, 32, 1, 2_097_152, 4)
RFC_9106_LOW_MEMORY = Parameters(Type.ID, 19, 16, 32, 3, 65_536, 4)
PRE_21_2 = Parameters(Type.ID, 19, 16, 16, 2, 102_400, 8)
CHEAPEST = Parameters(Type.ID, 19, 8, 4, 1, 8, 1)


def get_default_parameters() -> Parameters:
    return RFC_9106_LOW_MEMORY

"""Project-wide numeric constants and defaults.

Place all tunable "magic numbers" here so they are easy to discover and tweak.
"""
from typing import Tuple

# Sampling (table / cube placement)
DEFAULT_TABLE_Z_RANGE: Tuple[float, float] = (0.93, 1.3)
DEFAULT_CHECK_COLLISIONS: bool = True

# RRT / planning
DEFAULT_NUM_ITER: int = 200
DEFAULT_DISCRETISATION_STEPS: int = 100
DEFAULT_CHECK_EDGE_STEPS: int = 1000
EDGE_DISTANCE_TOL: float = 1e-3

# Visualization / interactive refresh defaults
DEFAULT_VIZ_DELAY: float = 0.01  # seconds to sleep after updating viz
SAMPLE_DISPLAY_EVERY: int = 1    # display every N samples when visualizing sampling
# Fraction of the table extents to use when creating a restricted sampling region
DEFAULT_SAMPLER_SHRINK: float = 0.3
# When sampling on the vertical plane, allow a small half-width (meters) perpendicular to the plane
DEFAULT_PLANE_HALF_WIDTH: float = 0.02

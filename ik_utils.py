"""IK helper utilities.

Provides a thin wrapper `call_ik` that calls an IK callback with a
backwards-compatible set of kwargs, times the call, and returns
((q, success), dt). The wrapper will inspect the callback signature and
only pass supported keyword arguments so older callbacks remain compatible.
"""
from typing import Callable, Any, Tuple
import time
import inspect


def call_ik(callback: Callable[..., Any], /, **kwargs) -> Tuple[Any, float]:
    """Call an IK callback while timing it and keeping backwards compatibility.

    The callback is expected to return a tuple like (q, success). `kwargs`
    may include: robot, qcurrent, cube, cubetarget, viz, max_attempts, viz_sleep.

    Returns:
        (result, dt) where `result` is whatever the callback returned and dt is
        the wall-time seconds spent in the call.
    """
    sig = inspect.signature(callback)
    # Filter kwargs to what the callback actually accepts
    call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    t0 = time.perf_counter()
    res = callback(**call_kwargs)
    dt = time.perf_counter() - t0
    return res, dt

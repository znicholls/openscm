"""
Utility functions for openscm.
"""
import datetime
import warnings
from typing import Any, Tuple, Union

from dateutil import relativedelta

OPENSCM_REFERENCE_TIME = datetime.datetime(1970, 1, 1, 0, 0, 0)


def ensure_input_is_tuple(inp: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """
    Return parameter as a tuple.

    Parameters
    ----------
    inp
        String or tuple to return as a tuple

    Returns
    -------
    Tuple[str, ...]
        A tuple with a single string `inp` if `inp` is a string, otherwise return `inp`
    """
    if isinstance(inp, str):
        if not getattr(ensure_input_is_tuple, "calls", 0):
            setattr(ensure_input_is_tuple, "calls", 1)
            warnings.warn("Converting input {} from string to tuple".format(inp))
        return (inp,)

    return inp


def convert_datetime_to_openscm_time(dt_in: datetime.datetime) -> int:
    """
    Convert a datetime.datetime instance to OpenSCM time i.e. seconds since OPENSCM_REFERENCE_TIME
    """
    return int((dt_in - OPENSCM_REFERENCE_TIME).total_seconds())


def convert_openscm_time_to_datetime(oscm_in: int) -> datetime.datetime:
    """Convert OpenSCM time to datetime.datetime"""
    # Need to cast to int as np.int64 from numpy arrays are unsupported
    return OPENSCM_REFERENCE_TIME + relativedelta.relativedelta(seconds=int(oscm_in))


def is_floatlike(f: Any) -> bool:
    """
    Check if input can be cast to a float

    This includes strings such as "6.03" which can be cast to a float

    Parameters
    ----------
    f
        Input

    Returns
    -------
    bool
        True if f can be cast to a float
    """
    if isinstance(f, (int, float)):
        return True

    try:
        float(f)
        return True
    except (TypeError, ValueError):
        return False


def round_to_nearest_jan_1(dtin):
    """
    Round a datetime to Jan 1st 00:00:00 of the nearest year
    thank you https://stackoverflow.com/a/48108115
    """
    dt_start_year = dtin.replace(
        month=1, day=1, minute=0, hour=0, second=0, microsecond=0
    )
    dt_half_year = dtin.replace(month=6, day=17)
    if dtin > dt_half_year:
        return dt_start_year + relativedelta.relativedelta(years=1)

    return dt_start_year

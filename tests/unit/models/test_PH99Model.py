import re


import pytest
import pint


from openscm.models import PH99Model


UREG = pint.UnitRegistry()
yr = 1*UREG.year
ONE_YEAR = yr.to("s")


@pytest.fixture(scope="function")
def ph_99():
    return PH99Model()

def test_step_time(ph_99):
    ph_99.time = [0*UREG.s, ONE_YEAR]
    ph_99.now = ph_99.time[0]

    ph_99.step_time()

    assert ph_99.now == ONE_YEAR

    # TODO: decide about error type
    error_msg = re.escape(
        "Cannot step time again as we are already at the last value in self.time"
    )
    with pytest.raises(AttributeError, match=error_msg):
        ph_99.step_time()

# test time has wrong unit error

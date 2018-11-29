import re
from unittest.mock import MagicMock


import numpy as np
import pytest
import pint


from openscm.models import PH99Model
from openscm.errors import OutOfTimeBoundsError


UREG = pint.UnitRegistry()
yr = 1*UREG.year
ONE_YEAR = yr.to("s")


@pytest.fixture(scope="function")
def ph_99():
    return PH99Model()

def test_step_time(ph_99):
    ph_99.time = np.array([0, ONE_YEAR.magnitude])*UREG.s
    ph_99.now = ph_99.time[0]

    ph_99.step_time()

    assert ph_99.now == ONE_YEAR


def test_step_time_out_of_bounds(ph_99):
    ph_99.time = np.array([0, ONE_YEAR.magnitude])*UREG.s
    ph_99.now = ph_99.time[-1]

    # TODO: decide about error type
    error_msg = re.escape(
        "Cannot step time again as we are already at the last value in self.time"
    )
    with pytest.raises(OutOfTimeBoundsError, match=error_msg):
        ph_99.step_time()


def test_time_idx(ph_99):
    ph_99.time = np.array([0, ONE_YEAR.magnitude])*UREG.s

    ph_99.now = ph_99.time[0]
    assert ph_99.time_idx == 0

    ph_99.now = ph_99.time[1]
    assert ph_99.time_idx == 1


def test_step(ph_99):
    ph_99.step_time = MagicMock()
    ph_99.step_cumulative_emissions = MagicMock()
    ph_99.step_concentrations = MagicMock()
    ph_99.step_temperature = MagicMock()

    ph_99.step()

    ph_99.step_time.assert_called_once()

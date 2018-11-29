import re
from unittest.mock import MagicMock


import numpy as np
import pytest
import pint


from openscm.models import PH99Model
from openscm.errors import OutOfTimeBoundsError, OverwriteError
from openscm.units import unit_registry


yr = 1 * unit_registry.year
ONE_YEAR = yr.to("s")


@pytest.fixture(scope="function")
def ph99():
    return PH99Model()


def test_step_time(ph99):
    ph99.time = np.array([0, ONE_YEAR.magnitude]) * unit_registry.s
    ph99.time_current = ph99.time[0]

    ph99._step_time()

    assert ph99.time_current == ONE_YEAR


def test_step_time_out_of_bounds(ph99):
    ph99.time = np.array([0, ONE_YEAR.magnitude]) * unit_registry.s
    ph99.time_current = ph99.time[-1]

    error_msg = re.escape(
        "Cannot step time again as we are already at the last value in self.time"
    )
    with pytest.raises(OutOfTimeBoundsError, match=error_msg):
        ph99._step_time()


def test_time_idx(ph99):
    ph99.time = np.array([0, ONE_YEAR.magnitude]) * unit_registry.s

    ph99.time_current = ph99.time[0]
    assert ph99.time_idx == 0

    ph99.time_current = ph99.time[1]
    assert ph99.time_idx == 1


def test_update_cumulative_emissions(ph99):
    ph99.cumulative_emissions = np.array([0, np.nan]) * unit_registry("GtC")
    ph99.emissions = np.array([0, 10]) * unit_registry("GtC/yr")
    ph99._check_update_overwrite = MagicMock()

    # TODO: work out how to do this mocking, this can't be the easy way
    ph99.time_idx = MagicMock(return_value=1)

    ph99._update_cumulative_emissions()
    # check if there's Pint testing which does this in one line for us
    np.testing.assert_allclose(ph99.cumulative_emissions.magnitude, np.array([0, 10]))
    assert ph99.cumulative_emissions.units == unit_registry("GtC")

    ph99._check_update_overwrite.assert_called_with("cumulative_emissions")


def test_check_update_overwrite(ph99):
    # TODO: work out how to do this mocking
    ph99.time_idx = MagicMock(return_value=1)

    ph99.cumulative_emissions = np.array([0, np.nan]) * unit_registry("GtC")
    # should pass without error
    ph99._check_update_overwrite("cumulative_emissions")

    ph99.cumulative_emissions = np.array([0, 10]) * unit_registry("GtC")
    error_msg = re.escape(
        "Stepping cumulative_emissions will overwrite existing data"
    )
    with pytest.raises(OverwriteError, match=error_msg):
        ph99._check_update_overwrite("cumulative_emissions")


def test_step(ph99):
    ph99._step_time = MagicMock()
    ph99._update_cumulative_emissions = MagicMock()
    ph99._update_concentrations = MagicMock()
    ph99._update_temperature = MagicMock()

    ph99.step()

    ph99._step_time.assert_called_once()
    ph99._update_cumulative_emissions.assert_called_once()
    ph99._update_concentrations.assert_called_once()
    ph99._update_temperature.assert_called_once()


# test that input arrays (time, emissions) are all same length, error if not
# test that units are correct, error if not
# test that initially all arrays are np.nan if not passed in

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
    ph99.emissions = np.array([2, 10]) * unit_registry("GtC/yr")
    ph99.timestep = 1 * unit_registry("yr")
    ph99._check_update_overwrite = MagicMock()

    # TODO: work out how to do this mocking, this can't be the easy way
    # ph99.time_idx = MagicMock(return_value=1)
    # temporary workaround
    ph99.time = np.array([0, 1])
    ph99.time_current = 1

    ph99._update_cumulative_emissions()

    # State variables are start of year values hence emissions only arrive in output
    # array in the next year. Hence add 10 GtC / yr * 1 yr to end cumulative emissions
    # array.
    expected_magnitude = np.array([0, 2])
    # check if there's Pint testing which does this in one line for us
    np.testing.assert_allclose(ph99.cumulative_emissions.magnitude, expected_magnitude)
    assert ph99.cumulative_emissions.units == unit_registry("GtC")

    ph99._check_update_overwrite.assert_called_with("cumulative_emissions")


def test_update_concentrations(ph99):
    ttimestep = 1 * unit_registry("yr")
    ph99.timestep = ttimestep

    tb = 1.5 * 10**-3 * unit_registry("ppm / (GtC * yr)")
    ph99.b = tb

    tcumulative_emissions = np.array([0, 2]) * unit_registry("GtC")
    ph99.cumulative_emissions = tcumulative_emissions

    tbeta = 0.46 * unit_registry("ppm/GtC")
    ph99.beta = tbeta

    temissions = np.array([2, 3]) * unit_registry("GtC/yr")
    ph99.emissions = temissions

    tsigma = 2.14 * 10**-2 * unit_registry("1/yr")
    ph99.sigma = tsigma

    tc1 = 289 * unit_registry("ppm")
    ph99.c1 = tc1

    tconcentrations = np.array([300, np.nan]) * unit_registry("ppm")
    ph99.concentrations = tconcentrations

    ph99._check_update_overwrite = MagicMock()

    # TODO: work out how to do this mocking, this can't be the easy way
    # ph99.time_idx = MagicMock(return_value=1)
    # temporary workaround
    ph99.time = np.array([0, 1])
    ph99.time_current = 1

    ph99._update_concentrations()
    # calculated from previous year values
    # TODO: check/discuss/explain. There is a more complex way of doing this gradient
    # calculation that means it's not actually forward differencing but rather
    # assuming constant emissions over the time period. It makes the calculation much
    # more complex (maybe even unsolvable...) for little reward in my opinion but we
    # should discuss/I should check.
    grad = (
        tb * tcumulative_emissions[0]
        + tbeta * temissions[0]
        - tsigma * (tconcentrations[0] - tc1)
    )
    expected_next_year_conc = tconcentrations[0] + grad * ttimestep

    expected_magnitude = np.array([
        tconcentrations[0].magnitude,
        expected_next_year_conc.magnitude
    ])
    # check if there's Pint testing which does this in one line for us
    np.testing.assert_allclose(ph99.concentrations.magnitude, expected_magnitude)
    assert ph99.concentrations.units == unit_registry("ppm")

    ph99._check_update_overwrite.assert_called_with("concentrations")


def test_update_temperatures(ph99):
    ttimestep = 1 * unit_registry("yr")
    ph99.timestep = ttimestep

    tmu = unit_registry.Quantity(8.6 * 10**-2, "degC/yr")
    ph99.mu = tmu

    tc1 = 289 * unit_registry("ppm")
    ph99.c1 = tc1

    tconcentrations = np.array([300, np.nan]) * unit_registry("ppm")
    ph99.concentrations = tconcentrations

    talpha = 1.6 * 10**-2 * unit_registry("1/yr")
    ph99.alpha = talpha

    tt1 = unit_registry.Quantity(14.7, "degC")
    ph99.t1 = tt1

    ttemperatures = unit_registry.Quantity(np.array([14.5, np.nan]), "degC")
    ph99.temperatures = ttemperatures

    ph99._check_update_overwrite = MagicMock()

    # TODO: work out how to do this mocking, this can't be the easy way
    # ph99.time_idx = MagicMock(return_value=1)
    # temporary workaround
    ph99.time = np.array([0, 1])
    ph99.time_current = 1

    ph99._update_temperatures()
    grad = (
        tmu * np.log(tconcentrations[0] / tc1)  # np.log is natural log
        - talpha * (ttemperatures[0] - tt1)
    )
    expected_next_year_temp = ttemperatures[0] + grad * ttimestep

    expected_magnitude = np.array([
        ttemperatures[0].magnitude,
        expected_next_year_temp.magnitude
    ])
    # check if there's Pint testing which does this in one line for us
    np.testing.assert_allclose(ph99.temperatures.magnitude, expected_magnitude)
    assert ph99.temperatures.units == unit_registry("degC")

    ph99._check_update_overwrite.assert_called_with("temperature")


def test_check_update_overwrite(ph99):
    # TODO: work out how to do this mocking
    # ph99.time_idx = MagicMock(return_value=1)
    # temporary workaround
    ph99.time = np.array([0, 1])
    ph99.time_current = 1

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
    ph99._update_temperatures = MagicMock()

    ph99.step()

    ph99._step_time.assert_called_once()
    ph99._update_cumulative_emissions.assert_called_once()
    ph99._update_concentrations.assert_called_once()
    ph99._update_temperatures.assert_called_once()


# test that input arrays (time, emissions) are all same length, error if not
# test that units are correct, error if not
# test that initially all arrays are np.nan if not passed in
# setting time should set time_current too

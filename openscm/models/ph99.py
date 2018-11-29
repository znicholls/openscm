"""
Questions as I write:
- Should the model do unit conversions internally or explode?

Decisions as I write:
- A model should take in a time array, and check that it matches with its internal timestep. Models shouldn't interpolate internally, that should be somewhere else in pre-processing. Models should also use regular timesteps. This means that months don't work (they vary in length and by year). Years also don't really make sense as they (strictly) vary in length. Hence most models should be working in days or minutes or hours or seconds (yes, strictly, these all also vary in length but those variations are sufficiently small not to matter). If people want to convert back to human calendars later, they can do so but that should also be a pre/post-processing step.
"""
import numpy as np


from ..units import unit_registry
from ..errors import OutOfTimeBoundsError, OverwriteError


class PH99Model(object):
    """Simple climate model first presented in Petschel-Held Climatic Change 1999

    This one box model projects global-mean CO2 concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of CO2 alone.

    Conventions:

    - All fluxes are time period averages and are assumed to be constant throughout
      the time period

    - All state variables are start of time period values


    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """

    # TODO: check if there is there a way to specify type when we define the variables
    # like in a function signature?
    time = np.array([np.nan]) * unit_registry("s")
    """`pint.Quantity` array: Time axis in seconds since 1970-1-1. Steps must match timestep"""
    # TODO: check if this is the right way to describe such a type
    # TODO: test timestep size checking

    time_current = np.nan * unit_registry("s")
    """int: Current time in seconds since 1970-1-1"""

    _yr = 1 * unit_registry("yr")
    timestep = _yr.to("s")
    """:obj:`pint.Quantity`: Size of timestep in seconds"""

    emissions = np.array([np.nan]) * unit_registry("GtC/s")
    """`pint.Quantity` array: Emissions of CO2 in GtC/s"""

    cumulative_emissions = np.array([np.nan]) * unit_registry("GtC")
    """`pint.Quantity` array: Cumulative emissions of CO2 in GtC"""

    concentrations = np.array([np.nan]) * unit_registry("ppm")
    """`pint.Quantity` array: Concentrations of CO2 in ppm"""

    temperatures = unit_registry.Quantity(np.array([np.nan]), "degC")
    """`pint.Quantity` array: Global-mean temperatures in degrees C"""

    b = 1.51 * 10**-3 * unit_registry("ppm / (GtC * yr)")
    """:obj:`pint.Quantity`: B parameter in ppm / (GtC yr)"""

    beta = 0.47 * unit_registry("ppm/GtC")
    """:obj:`pint.Quantity`: beta parameter in ppm / GtC

    This is the fraction of emissions which impact the carbon cycle.
    """

    sigma = 2.15 * 10**-2 * unit_registry("1/yr")
    """:obj:`pint.Quantity`: sigma parameter in yr^-1

    The characteristic response time of the carbon cycle.
    """
    # TODO: check if you can put latex in docstrings

    c1 = 290 * unit_registry("ppm")
    """:obj:`pint.Quantity`: C1 parameter in ppm

    The pre-industrial CO2 concentration.
    """

    mu = unit_registry.Quantity(8.7 * 10**-2, "degC/yr")
    """:obj:`pint.Quantity`: mu parameter in degrees C / yr

    This is like a scaling factor of the radiative forcing due to CO2 but has
    different units as it is used directly in a temperature response equation rather
    than an energy balance equation.
    """

    alpha = 1.7 * 10**-2 * unit_registry("1/yr")
    """:obj:`pint.Quantity`: alpha parameter in yr^-1

    The characteristic response time of global-mean temperatures.
    """

    t1 = unit_registry.Quantity(14.6, "degC")
    """:obj:`pint.Quantity`: T1 parameter in degrees C

    The pre-industrial global-mean temperature.
    """

    @property
    def time_idx(self):
        if np.isnan(self.time_current):
            return None

        return np.argmax(self.time == self.time_current)

    def run(self, until=(2500-1970)*365*24*60*60*unit_registry("s"), restart=False) -> None:
        """Run the model

        Parameters
        ----------
        until: :obj:`pint.Quantity`
            Time to run until. Default value is 2500.

        restart: bool
            If True, run the model from the first timestep rather than from the value
            of `self.time_current`. This will overwrite any values which have already
            been calculated.
        """
        # super nice that we don't have to write type in docstring when the type is in the function signature
        pass

    def step(self) -> None:
        """Step the model forward to the next point in time"""
        self._step_time()
        self._update_cumulative_emissions()
        self._update_concentrations()
        self._update_temperatures()

    def _step_time(self) -> None:
        try:
            self.time_current = self.time[self.time_idx + 1]
        except IndexError:
            raise OutOfTimeBoundsError(
                "Cannot step time again as we are already at the last value in "
                "self.time"
            )

    def _update_cumulative_emissions(self) -> None:
        """Update the cumulative emissions"""
        self._check_update_overwrite("cumulative_emissions")
        self.cumulative_emissions[self.time_idx] = (
            self.cumulative_emissions[self.time_idx - 1]
            + self.emissions[self.time_idx - 1] * self.timestep
        )

    def _update_concentrations(self) -> None:
        """Update the concentrations"""
        self._check_update_overwrite("concentrations")
        dcdt = (
            self.b * self.cumulative_emissions[self.time_idx - 1]
            + self.beta * self.emissions[self.time_idx - 1]
            - self.sigma * (self.concentrations[self.time_idx - 1] - self.c1)
        )
        self.concentrations[self.time_idx] = (
            self.concentrations[self.time_idx - 1]
            + dcdt * self.timestep
        )

    def _update_temperatures(self) -> None:
        """Update the concentrations"""
        self._check_update_overwrite("temperature")
        dtdt = (
            self.mu * np.log(self.concentrations[self.time_idx - 1] / self.c1)
            - self.alpha * (self.temperatures[self.time_idx - 1] - self.t1)
        )
        self.temperatures[self.time_idx] = (
            self.temperatures[self.time_idx - 1]
            + dtdt * self.timestep
        )

    def _check_update_overwrite(self, attribute_to_check) -> None:
        """Check if updating the given array will overwrite existing data

        Parameters
        ----------
        attribute_to_check: str
            The attribute of self to check.

        Raises
        ------
        OverwriteError
            If updating the array stored in `attribute_to_check` will overwrite data
            which has already been calculated.
        """
        array_to_check = self.__getattribute__(attribute_to_check)
        if not np.isnan(array_to_check[self.time_idx]):
            raise OverwriteError(
                "Stepping {} will overwrite existing data".format(attribute_to_check)
            )

"""
Questions as I write:
- Should stepping a model forward do the timesteps given in

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
        self._update_temperature()

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
            + self.emissions[self.time_idx] * self.timestep
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

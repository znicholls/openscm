"""
Questions as I write:
- Should stepping a model forward do the timesteps given in

Decisions as I write:
- A model should take in a time array, and check that it matches with its internal timestep. Models shouldn't interpolate internally, that should be somewhere else in pre-processing. Models should also use regular timesteps. This means that months don't work (they vary in length and by year). Years also don't really make sense as they (strictly) vary in length. Hence most models should be working in days or minutes or hours or seconds (yes, strictly, these all also vary in length but those variations are sufficiently small not to matter). If people want to convert back to human calendars later, they can do so but that should also be a pre/post-processing step.
"""
import numpy as np


from ..units import unit_registry
from ..errors import OutOfTimeBoundsError


class PH99Model(object):
    """Simple climate model first presented in Petschel-Held Climatic Change 1999

    This one box model projects global-mean CO2 concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of CO2 alone.

    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """

    # TODO: decide whether we want int or float for time
    # TODO: check if there is there a way to specify type when we define the variables
    # like in a function signature?
    time = [
        np.nan
    ] * unit_registry.s  # default has to be None, anything else doesn't make sense
    """array of `pint.Quantity`: time axis in seconds since 1970-1-1"""

    time_current = None  # default has to be None, anything else doesn't make sense
    """int: Current time in seconds since 1970-1-1"""

    @property
    def time_idx(self):
        if self.time_current is None:
            return None

        return np.argmax(self.time == self.time_current)

    def run(self, restart: bool) -> None:
        """Run the model

        Parameters
        ----------
        restart
            If True, run the model from the first timestep rather than from the value of self.time_current. This will overwrite any values which have already been
            calculated.
        """
        # super nice that we don't have to write type in docstring when the type is in the function signature
        pass

    def step(self) -> None:
        """Step the model forward to the next point in time"""
        self._step_time()
        self._step_cumulative_emissions()
        self._step_concentrations()
        self._step_temperature()

    def _step_time(self) -> None:
        try:
            self.time_current = self.time[self.time_idx + 1]
        except IndexError:
            raise OutOfTimeBoundsError(
                "Cannot step time again as we are already at the last value in "
                "self.time"
            )

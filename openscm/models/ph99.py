"""
Questions as I write:
- Should stepping a model forward do the timesteps given in

Decisions as I write:
- A model should take in a time array, and check that it matches with its internal timestep. Models shouldn't interpolate internally, that should be somewhere else in pre-processing. Models should also use regular timesteps. This means that months don't work (they vary in length and by year). Years also don't really make sense as they (strictly) vary in length. Hence most models should be working in days or minutes or hours or seconds (yes, strictly, these all also vary in length but those variations are sufficiently small not to matter). If people want to convert back to human calendars later, they can do so but that should also be a pre/post-processing step.
"""
class PH99Model(object):
    """Simple climate model first presented in Petschel-Held Climatic Change 1999

    This one box model projects global-mean CO2 concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of CO2 alone.

    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """
    now = None  # default has to be None, anything else doesn't make sense
    """int: Current time in seconds since 1970-1-1"""
    # TODO: decide whether we want int or float for time


    def run(self, restart: bool) -> None:
        """Run the model

        Parameters
        ----------
        restart
            If True, run the model from the first timestep rather than from the value of self.now. This will overwrite any values which have already been
            calculated.
        """
        # super nice that we don't have to write type in docstring when the type is in the function signature
        pass

    def step(self) -> None:
        """Step the model forward to the next point in time"""

        # update current time
        # step cumulative emissions
        # step concentrations
        # step temperature
        pass


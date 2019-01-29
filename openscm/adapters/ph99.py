from copy import deepcopy


import numpy as np


from ..core import ParameterSet
from ..internal import Adapter
from ..models import PH99Model
from ..units import unit_registry
from ..errors import ParameterWrittenError

"""
Questions as I write:
- how to cache model instances for adapters, doesn't really matter here but matters more models like MAGICC which are expensive to spin up
"""

ONE_YEAR_INT = int(1*unit_registry("yr").to("s").magnitude)


class PH99(Adapter):
    """Adapter for the simple climate model first presented in Petschel-Held Climatic Change 1999

    This one box model projects global-mean CO2 concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of CO2 alone.

    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """

    def __init__(self):
        pass

    def initialize(self) -> None:
        """
        Initialize the model.
        """
        pass

    # do I need to copy the output or is that inherited from superclass?
    def run(self, **kwargs) -> None:
        self.model.time_current = self.model.time_start
        # I need to add a setter which sets other arrays based on length of emissions
        # and does re-setting etc.
        # temporary workaround
        initialiser = np.nan * np.zeros_like(self.model.emissions.magnitude)

        cumulative_emissions_init = deepcopy(initialiser)
        cumulative_emissions_init[0] = 0
        self.model.cumulative_emissions = unit_registry.Quantity(
            cumulative_emissions_init,
            "GtC"
        )

        concentrations_init = deepcopy(initialiser)
        concentrations_init[0] = 290
        self.model.concentrations = unit_registry.Quantity(
            concentrations_init,
            "ppm"
        )

        temperatures_init = deepcopy(initialiser)
        temperatures_init[0] = 14.6
        self.model.temperatures = unit_registry.Quantity(
            temperatures_init,
            "degC"
        )

        self.model.run(**kwargs)

        results = ParameterSet()
        results.start_time = self.model.time_start.magnitude
        results.period_length = self.model.timestep.magnitude

        for att in dir(self.model):
            # all time parameters captured in parameterset output
            if not att.startswith(("_", "time")):
                value = getattr(self.model, att)
                if callable(value):
                    continue

                name = self._get_openscm_name(att)
                magnitude = value.magnitude

                if isinstance(magnitude, np.ndarray):
                    view = results.get_writable_timeseries_view(
                        (name,),  # shouldn't need trailing comma
                        (),  # make this so World actually works
                        str(value.units),
                        self.model.time_start.magnitude,
                        self.model.timestep.magnitude,
                    )
                    view.set_series(magnitude)
                else:
                    view = results.get_writable_scalar_view(
                        (name,),  # shouldn't need trailing comma
                        (),  # make this so World actually works
                        str(value.units)
                    )
                    view.set(magnitude)

        return results

    def _get_openscm_name(self, name):
        mappings = {
            "concentrations": "Atmospheric Concentrations|CO2",
            "cumulative_emissions": "Cumulative Emissions|CO2",
            "emissions": "Emissions|CO2",
            "temperatures": "Surface Temperature",
        }
        try:
            return mappings[name]
        except KeyError:
            return name

    def step(self) -> None:
        self.model.step()

    def setup_scenario(self, parameters: ParameterSet, start_time: int, period_length: int = ONE_YEAR_INT) -> None:
        """
        Setup the model to run a given scenario.
        """
        self.model = PH99Model(time_start=start_time * unit_registry("s"))
        self.model.timestep = period_length * unit_registry("s")

        units = "Gt C / s"
        # TODO: get aggregated read working...
        emms_co2_fossil = parameters.get_timeseries_view(
            ("Emissions", "CO2", "MAGICC Fossil and Industrial"), (),
            units,
            start_time,
            period_length,
        )

        emms_co2_afolu = parameters.get_timeseries_view(
            ("Emissions", "CO2", "MAGICC AFOLU"), (),
            units,
            start_time,
            period_length,
        )

        emms_co2_afolu.length
        emms_co2 = emms_co2_fossil.get_series() + emms_co2_afolu.get_series()

        self.model.emissions = emms_co2 * unit_registry(units)

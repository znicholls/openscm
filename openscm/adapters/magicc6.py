from datetime import datetime


import numpy as np
import pandas as pd
import pymagicc
from pymagicc.io import MAGICCData
from pymagicc.definitions import (
    MAGICC7_EMISSIONS_UNITS,
    convert_magicc7_to_openscm_variables,
)


from ..core import ParameterSet
from ..units import unit_registry
from ..internal import Adapter
from ..utils import convert_datetime_to_openscm_time


ONE_YEAR_INT = int(1*unit_registry("yr").to("s").magnitude)


class MAGICC6(Adapter):
    def __init__(self):
        self.magicc = pymagicc.MAGICC6()

    def initialize(self) -> None:
        """
        Initialize the model.
        """
        # TODO: make this able to use permanent magicc folders (rather than temporary)
        self.magicc.create_copy()

    def run(self, **kwargs) -> None:
        """
        Run the model over the full time range.
        """
        # in here is where all the switches about conc driven, emissions driven
        # etc. would be processed
        drivers = self.get_pymagicc_df()
        results = self.magicc.run(drivers, **kwargs)

        return results

    def setup_scenario(self, parameters: ParameterSet, start_time: int, period_length: int = ONE_YEAR_INT):
        pymagicc_df = self.get_pymagicc_df(parameters)
        assert period_length == ONE_YEAR_INT, "non year timesteps not currently available"
        pass

    def get_pymagicc_df(self, parameters: ParameterSet) -> pymagicc.io.MAGICCData:
        ONE_YEAR_INT
        scen_file_emissions_units = MAGICC7_EMISSIONS_UNITS[
            MAGICC7_EMISSIONS_UNITS["part_of_scenfile_with_emissions_code_1"]
        ]  # can only convert to SCEN file for now
        region = ("World", )  # TODO: remove hard coding
        stime = convert_datetime_to_openscm_time(datetime(1765, 7, 12, 0, 0, 0))  # TODO: remove hard coding

        emms_df = []
        for (variable, unit), df in scen_file_emissions_units.groupby(["magicc_variable", "emissions_unit"]):
            variable_well_defined = "{}_EMIS".format(variable)
            openscm_variable = convert_magicc7_to_openscm_variables(variable_well_defined)

            emms_view = parameters.get_timeseries_view(
                tuple(openscm_variable.split("|")),  # TODO: remove hard coding
                region,
                unit,
                stime,
                ONE_YEAR_INT,
            )

            values = emms_view.get_series()
            assert len(values) == 736
            time = np.arange(1765, 2501)  # TODO: remove hard coding
            emms_df.append(pd.DataFrame({
                "variable": openscm_variable,
                "todo": "SET",
                "unit": unit,
                "region": "|".join(region),  # TODO: remove hard coding
                "time": time,
                "value": values,
            }))

        emms_df = pd.concat(emms_df)
        mdata = MAGICCData(emms_df)

        return mdata

    def step(self) -> None:
        """
        Do a single time step.
        """
        raise NotImplementedError

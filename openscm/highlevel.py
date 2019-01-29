"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
from datetime import datetime, timedelta
from dateutil import parser


import numpy as np
import pandas as pd
from pyam import IamDataFrame


from .core import Core, ParameterSet
from .utils import convert_datetime_to_openscm_time
from .units import unit_registry


class OpenSCM(Core):
    """
    High-level OpenSCM class.

    Represents model runs with a particular simple climate model.
    """

    pass

class ScmDataFrameBase(IamDataFrame):
    """This base is the class other libraries can subclass

    Having such a subclass avoids a potential circularity where e.g. openscm imports ScmDataFrame as well as Pymagicc, but Pymagicc wants to import ScmDataFrame and hence to try and import ScmDataFrame you have to import ScmDataFrame itself (hence the circularity).
    """
    def _format_datetime_col(self):
        if isinstance(self.data["time"].iloc[0], str):
            def convert_str_to_datetime(inp):
                return parser.parse(inp)

            self.data["time"] = self.data["time"].apply(convert_str_to_datetime)

        not_datetime = [not isinstance(x, datetime) for x in self.data["time"]]
        if any(not_datetime):
            bad_values = self.data[not_datetime]["time"]
            error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(bad_values)
            raise ValueError(error_msg)

    def append(self, other, **kwargs):
        if not isinstance(other, OpenSCMDataFrame):
            other = OpenSCMDataFrame(other, **kwargs)

        super().append(other, inplace=True)


class ScmDataFrame(ScmDataFrameBase):
    """OpenSCM's custom DataFrame implementation.

    The ScmDataFrame wraps around `pyam`'s IamDataFrame, which itself wraps around Pandas.

    The ScmDataFrame provides a number of diagnostic features (including validation
    of data, completeness of variables provided, running of simple climate models)
    as well as a number of visualization and plotting tools.
    """
    pass


def convert_openscm_df_to_parameter_set(openscm_df):
    # make internal
    ONE_YEAR_INT = int(1*unit_registry("yr").to("s").magnitude)

    parameter_set = ParameterSet()
    for (variable, region), df in openscm_df.data.groupby(["variable", "region"]):
        df = df.sort_values("time")
        variable_openscm = tuple(variable.split("|"))  # TODO: remove hard coding
        region_openscm = tuple(region.split("|"))
        # TODO: discuss why this is and make handling better
        assert region_openscm[0] == "World"
        region_openscm = region_openscm[1:] if len(region_openscm) > 1 else ()
        unit = df.unit.unique()
        assert len(unit) == 1, "emissions timeseries should all be in one unit"  # TODO: remove this restriction
        unit = unit[0]
        syr = df.time.min().year
        assert syr == 1765, "have not considered cases other than the RCPs yet"  # TODO: remove this restriction
        eyr = df.time.max().year
        assert eyr == 2500, "have not considered cases other than the RCPs yet"  # TODO: remove this restriction
        assert len(df) == 736, "have not considered cases other than the RCPs read in by pymagicc yet"  # TODO: remove this restriction
        tstep = ONE_YEAR_INT  # having passed all above, can safely assume this [TODO: remove this assumption]

        emms_view = parameter_set.get_writable_timeseries_view(
            variable_openscm,
            region_openscm,
            unit,
            convert_datetime_to_openscm_time(datetime(syr, 1, 1, 0, 0, 0)),
            tstep
        )
        emms_view.set_series(df["value"].values)

    return parameter_set


def convert_parameter_set_to_openscmdf(
    parameter_set,
    climate_model,
    scenario,
    model="unspecified",
):
    for key, value in parameter_set._world._parameters.items():
        values = value._data
        if isinstance(values, np.ndarray):
            time_length = len(values)
            time = [
                parameter_set.start_time + parameter_set.period_length * i
                for i in range(len(values))
            ]
            time = [
                datetime(1970, 1, 1, 0, 0, 0) + timedelta(seconds=j)
                for j in time
            ]
            break

    metadata = {
        "climate_model": [climate_model] * len(time),
        "scenario": [scenario] * len(time),
        "model": [model] * len(time),
    }

    dataframes = []
    for key, value in parameter_set._world._parameters.items():
        variable = value.info.name
        region = value.info.region if value.info.region else "World"  # TODO: fix this
        unit = value.info.unit
        tdf = {
            **metadata,
            "variable": [variable] * len(time),
            "unit": [unit] * len(time),
            "region": [region] * len(time),
            "time": time,
        }
        values = value._data
        if isinstance(values, float):
            tdf["value"] = [values] * len(time)
        else:
            assert len(values) == time_length
            tdf["value"] = values

        dataframes.append(pd.DataFrame(tdf))

    return OpenSCMDataFrame(pd.concat(dataframes))

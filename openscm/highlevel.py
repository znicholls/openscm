"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
from datetime import datetime, timedelta
from dateutil import parser


import numpy as np
import pandas as pd
from progressbar import progressbar


from .core import Core, ParameterSet
from .utils import convert_datetime_to_openscm_time
from .units import unit_registry
from .adapters import get_adapter
from .scmdataframebase import ScmDataFrameBase


class OpenSCM(Core):
    """
    High-level OpenSCM class.

    Represents model runs with a particular simple climate model.
    """

    pass


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
    # TODO: remove hard coding here
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
    # how do you keep track of units in metadata.. may have to get units in pandas
    # deployed
    for key, value in parameter_set._world._parameters.items():
        values = value._data
        variable = value.info.name
        if isinstance(values, float):
            metadata["{} ({})".format(variable, value.info.unit)] = values

    dataframes = []
    for key, value in parameter_set._world._parameters.items():
        values = value._data
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
        if not isinstance(values, float):
            assert len(values) == time_length
            tdf["value"] = values

            dataframes.append(pd.DataFrame(tdf))

    result = ScmDataFrame(pd.concat(dataframes))

    return result


def convert_config_dict_to_parameter_set(config):
    assert isinstance(config, dict)
    parameters = ParameterSet()
    for key, (region, value) in config.items():
        # TODO: remove need for trailing comma
        region = () if region == "World" else (region, )  # TODO: remove this
        view = parameters.get_writable_scalar_view(
            (key,),  # TODO: remove need for trailing comma
            region,
            str(value.units),
        )
        view.set(value.magnitude)

    return parameters

def run(drivers, model_configurations):
    assert isinstance(model_configurations, dict), "model_configurations must be a dictionary"
    for climate_model, configurations in model_configurations.items():
        print("running {}\n".format(climate_model))
        runner = get_adapter(climate_model)()
        runner.initialize()
        for (scenario, model), sdf in drivers.data.groupby(["scenario", "model"]):
            print("running {}".format(scenario))
            parameter_set_scenario = convert_openscm_df_to_parameter_set(
                ScmDataFrame(sdf.copy())
            )
            runner.setup_scenario(
                parameters=parameter_set_scenario,
                start_time=convert_datetime_to_openscm_time(sdf["time"].min()),
            )
            for i, config in progressbar(enumerate(configurations)):
                parameter_set_config = convert_config_dict_to_parameter_set(config)
                config_results = runner.run(parameter_set_config)

                config_results = convert_parameter_set_to_openscmdf(
                    config_results,
                    climate_model,
                    scenario,
                    model=model,
                )
                try:
                    results.append(
                        config_results,
                        inplace=True,
                        ignore_meta_conflict=True,  # TODO: make meta_idx class specific in IamDataFrame
                    )
                except NameError:
                    results = config_results

    return results

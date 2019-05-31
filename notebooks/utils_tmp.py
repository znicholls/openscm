# TODO's whilst writing:

# - expose `unit_registry` publicly, seems easiest way to do things?
# - fix warning when doing `to_core` calls with `climate_model` column set
# - work out usefulness/otherwise of region in scalar views
# - convenience method for setting start/stop time sensibly
# - fix conversion of core to scmdataframe and back (remove hard-coding of parameter type)
# - add method to get point timeseries from average and vice versa
# - add get unit for variable method
# - decide if time points can be non-int

import datetime as dt

import numpy as np

from openscm.adapters import load_adapter
from openscm.core import ParameterSet
from openscm.scmdataframe import ScmDataFrame, df_append
from openscm.timeseries_converter import ParameterType
from openscm.utils import convert_datetime_to_openscm_time, convert_openscm_time_to_datetime

def write_scenario_df_to_parameterset(sdf, paraset):
    # nasty hack, we really need a `overwrite_values(other_parameter_set)` method
    # on `ParameterSet` if we want to be able to have the parameters persist
    for v in [("Emissions", "CO2", "MAGICC AFOLU"), ("Emissions", "CO2", "MAGICC Fossil and Industrial")]:
        region="World"
        fdf = sdf.filter(
            variable=sdf.data_hierarchy_separator.join(v),
            region=region
        )
        units = fdf["unit"].unique()
        assert len(units) == 1
        units = units[0]
        time_bounds = fdf.time_points
        time_bounds = np.concatenate([
            time_bounds,
            [2*time_bounds[-1] - time_bounds[-2]]
        ])

        paraset.get_writable_timeseries_view(
            v,
            (region,),
            units,
            time_bounds,
            ParameterType.AVERAGE_TIMESERIES
        ).set(fdf.values.squeeze())


def write_config_to_parameterset(cfg, paraset):
    for name, value in cfg.items():
        paraset.get_writable_scalar_view(
            name,
            ("World",),  # hard-coded!!
            str(value.units)
        ).set(value.magnitude)


def convert_output_parameterset_cfg_to_scmdataframe(paraset, cfg, climate_model, model, scenario, region="World"):
    # hack around broken functions
    out_times = [
        convert_datetime_to_openscm_time(dt.datetime(y, 1, 1)) for y in range(1950, 2050)
    ]
    temperature_units = "delta_degC"
    temperature_values = paraset.get_timeseries_view(
        ("Surface Temperature",),
        (region,),
        temperature_units,
        out_times,
        ParameterType.POINT_TIMESERIES,
    ).get()

    conc_units = "ppm"
    conc_values = paraset.get_timeseries_view(
        ("Atmospheric Concentrations", "CO2"),
        (region,),
        conc_units,
        out_times,
        ParameterType.POINT_TIMESERIES,
    ).get()

    res_here = ScmDataFrame(
        data=np.vstack([temperature_values, conc_values]).T,
        index=[convert_openscm_time_to_datetime(t) for t in out_times],
        columns={
            "climate_model": climate_model,
            "model": model,
            "scenario": scenario,
            "unit": [temperature_units, conc_units],
            "variable": ["Surface Temperature", "Atmospheric Concentrations|CO2"],
            "region": region,
        }
    )
    for name, value in cfg.items():
        label = "{} ({})".format(
            "|".join(name),
            value.units
        )
        res_here.set_meta(value.magnitude, label)

    return res_here

def run(scens, climate_model_cfgs):
    results = []
    for cm, cfgs in climate_model_cfgs.items():
        input_paraset = ParameterSet()
        output_paraset = ParameterSet()
        adapter = load_adapter(cm)(input_paraset, output_paraset)

        for (model, scenario), df in scens._meta.groupby(["model", "scenario"]):
            print(model)
            print(scenario)
            mod_scen_scmdf = scens.filter(model=model, scenario=scenario)
            mod_scen_scmdf.set_meta(cm, name="climate_model")

            # will have new interface in future
            write_scenario_df_to_parameterset(mod_scen_scmdf, input_paraset)

            adapter.initialize_model_input()

            for cfg in cfgs:
                # will have new interface in future
                write_config_to_parameterset(cfg, adapter._parameters)

                adapter.initialize_run_parameters()
                adapter.reset()  # right order?
                adapter.run()

                # will have new interface in future
                res_here = convert_output_parameterset_cfg_to_scmdataframe(
                    output_paraset,
                    cfg,
                    climate_model=cm,
                    model=model,
                    scenario=scenario,
                )

                results.append(res_here)


    results = df_append(results)

    return results

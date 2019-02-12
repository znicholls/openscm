import warnings
from collections import OrderedDict
import copy
import inspect


import numpy as np
from fair.forward import fair_scm


from ..internal import Adapter
from ..models import PH99Model
from ..core import Core, ParameterSet
from ..units import unit_registry
from ..errors import NotAnScmParameterError
from ..parameters import _Parameter, ParameterType
from ..constants import ONE_YEAR_IN_S_INTEGER
from ..utils import (
    convert_openscm_time_to_datetime,
    round_to_nearest_year
)


_emissions_units_index_fair = OrderedDict({
    ("time",): "yr",  # ignored for now, just for indexing
    ("CO2", "MAGICC Fossil and Industrial"): "GtC / yr",
    ("CO2", "MAGICC AFOLU"): "GtC / yr",
    ("CH4",): "MtCH4 / yr",
    ("N2O",): "MtN2ON / yr",
    ("SOx",): "MtS / yr",
    ("CO",): "MtCO / yr",
    ("NMVOC",): "MtNMVOC / yr",
    ("NOx",): "MtN / yr",
    ("BC",): "MtBC / yr",
    ("OC",): "MtOC / yr",
    ("NH3",): "MtNH3 / yr",
    ("CF4",): "MtCF4 / yr",
    ("C2F6",): "MtC2F6 / yr",
    ("C6F14",): "MtC6F14 / yr",
    ("HFC23",): "ktHFC23 / yr",
    ("HFC32",): "ktHFC32 / yr",
    ("HFC4310",): "ktHFC4310 / yr",
    ("HFC125",): "ktHFC125 / yr",
    ("HFC134a",): "ktHFC134a / yr",
    ("HFC143a",): "ktHFC143a / yr",
    ("HFC227ea",): "ktHFC227ea / yr",
    ("HFC245fa",): "ktHFC245fa / yr",
    ("SF6",): "ktSF6 / yr",
    ("CFC11",): "ktCFC11 / yr",
    ("CFC12",): "ktCFC12 / yr",
    ("CFC113",): "ktCFC113 / yr",
    ("CFC114",): "ktCFC114 / yr",
    ("CFC115",): "ktCFC115 / yr",
    ("CCl4",): "ktCCl4 / yr",
    ("CH3CCl3",): "ktCH3CCl3 / yr",
    ("HCFC22",): "ktHCFC22 / yr",
    ("HCFC141b",): "ktHCFC141b / yr",
    ("HCFC142b",): "ktHCFC142b / yr",
    ("Halon1211",): "ktHalon1211 / yr",
    ("Halon1202",): "ktHalon1202 / yr",
    ("Halon1301",): "ktHalon1301 / yr",
    ("Halon2402",): "ktHalon2402 / yr",
    ("CH3Br",): "ktCH3Br / yr",
    ("CH3Cl",): "ktCH3Cl / yr",
})
for i, (k, v) in enumerate(_emissions_units_index_fair.items()):
    _emissions_units_index_fair[k] = {"unit": v, "index": i}

_concentrations_units_index_fair = OrderedDict({
    ("CO2",): "ppm",
    ("CH4",): "ppb",
    ("N2O",): "ppb",
    ("CF4",): "ppt",
    ("C2F6",): "ppt",
    ("C6F14",): "ppt",
    ("HFC23",): "ppt",
    ("HFC32",): "ppt",
    ("HFC4310",): "ppt",
    ("HFC125",): "ppt",
    ("HFC134a",): "ppt",
    ("HFC143a",): "ppt",
    ("HFC227ea",): "ppt",
    ("HFC245fa",): "ppt",
    ("SF6",): "ppt",
    ("CFC11",): "ppt",
    ("CFC12",): "ppt",
    ("CFC113",): "ppt",
    ("CFC114",): "ppt",
    ("CFC115",): "ppt",
    ("CCl4",): "ppt",
    ("CH3CCl3",): "ppt",
    ("HCFC22",): "ppt",
    ("HCFC141b",): "ppt",
    ("HCFC142b",): "ppt",
    ("Halon1211",): "ppt",
    ("Halon1202",): "ppt",
    ("Halon1301",): "ppt",
    ("Halon2402",): "ppt",
    ("CH3Br",): "ppt",
    ("CH3Cl",): "ppt",
})
for i, (k, v) in enumerate(_concentrations_units_index_fair.items()):
    _concentrations_units_index_fair[k] = {"unit": v, "index": i}

_radiative_forcing_units_index_fair = OrderedDict({
    ("CO2",): "W / m^2",
    ("CH4",): "W / m^2",
    ("N2O",): "W / m^2",
    ("GHG (need a better name)",): "W / m^2",
    ("O3|Tropospheric",): "W / m^2",
    ("O3|Stratospheric",): "W / m^2",
    ("Stratospheric Water Vapour from CH4 Oxidation",): "W / m^2",
    ("Contrails",): "W / m^2",
    ("Aerosols",): "W / m^2",
    ("Black Carbon on Snow",): "W / m^2",
    ("Land Use Change",): "W / m^2",
    ("Volcanic",): "W / m^2",
    ("Solar",): "W / m^2",
})
for i, (k, v) in enumerate(_radiative_forcing_units_index_fair.items()):
    _radiative_forcing_units_index_fair[k] = {"unit": v, "index": i}

# how to do this intelligently and scalably?
_map = {"rf2xco2": "F2x"}
_openscm_para_fair_mapping = copy.deepcopy(_map)
for k, v in _map.items():
    _openscm_para_fair_mapping[k] = v

_parameters_fair = {
    "ecs": {"type": ParameterType.SCALAR, "unit": "kelvin"},
    "F2x": {"type": ParameterType.SCALAR, "unit": "W / m^2"},
    "r0": {"type": ParameterType.SCALAR, "unit": "yrs"},
}

class FAIR(Adapter):
    """Adapter for FaIR, https://github.com/OMS-NetZero/FAIR"""
    def __init__(self):
        self.model = None
        self.name = "FaIR"
        self._drivers = None
        self._config = None
        self._run_start = np.nan
        self._timestep = np.nan
        super().__init__()

    def initialize(self):
        super().initialize()
        self.model = fair_scm

    def set_drivers(self, core: Core) -> None:
        self._run_start = core.start_time
        try:
            self._timestep = core.parameters._root._parameters["Emissions"]._children["BC"]._info._timeframe.period_length
            for cp in core.parameters._root._parameters["Emissions"]._children.values():
                self._set_drivers_from_child_para(core, cp)
        except KeyError:
            raise NotImplementedError("Not ready for non-RCP emissions runs yet")

    def _set_drivers_from_child_para(self, core: Core, parameter: _Parameter):
        if parameter._children:
            for cp in parameter._children.values():
                # this will break at some point but should be ok for now
                self._set_drivers_from_child_para(core, cp)
            return

        pview = core.parameters.get_timeseries_view(
            parameter.full_name,
            ("World",),  # FaIR is always global
            _emissions_units_index_fair[parameter.full_name[1:]]["unit"],
            self._run_start,
            self._timestep,
        )
        if self._drivers is None:
            # hack central
            times = pview.get_times()
            nt = len(times)
            self._drivers = {"emissions": np.zeros((nt, len(_emissions_units_index_fair.keys())))}
            self._drivers["emissions"][:, 0] = np.array([round_to_nearest_year(convert_openscm_time_to_datetime(int(t))).year for t in times])


        self._drivers["emissions"][
            :,
            _emissions_units_index_fair[parameter.full_name[1:]]["index"]
        ] = pview.get_series()

    def set_config(self, parameters: ParameterSet) -> None:
        super().set_config(parameters)
        self._config = {}  # might need to be more careful with this in future...
        # this is where a loop over all parameters method would be useful rather
        # than needing to access private attributes
        for key, value in parameters._root._parameters.items():
            self._set_model_parameter(key, value)

    def _set_model_parameter(self, para_name, value):
        modval = value._data * unit_registry(value.info.unit)
        try:
            magnitude = modval.to(_parameters_fair[para_name]["unit"]).magnitude
            if para_name in ("ecs", "tcr"):
                if para_name == "ecs":
                    self._config["ecs"] = magnitude
                    if "tcr" not in self._config:
                        signature = inspect.signature(self.model)
                        self._config["tcr"] = signature.parameters["tcrecs"].default[0]
                else:
                    self._config["tcr"] = magnitude
                    if "ecs" not in self._config:
                        signature = inspect.signature(self.model)
                        self._config["ecs"] = signature.parameters["tcrecs"].default[1]
                self._config["tcrecs"] = np.array([
                    self._config["tcr"], self._config["ecs"]
                ])
            else:
                self._config[para_name] = magnitude
        except KeyError:
            try:
                para_name = _openscm_para_fair_mapping[para_name]
                magnitude = modval.to(_parameters_fair[para_name]["unit"]).magnitude
                self._config[para_name] = magnitude
            except KeyError:
                raise NotAnScmParameterError("{} is not a {} parameter".format(para_name, self.name))

    def run(self) -> None:
        config = {
            k: v for k, v in self._config.items() if k not in ("ecs", "tcr")
        }

        concs, forcing, temperature = fair_scm(**self._drivers, **config)

        results = Core(
            "PH99",
            self._run_start,
            12  # I don't think this is used anywhere
        )

        results.period_length = self._timestep

        results.parameters.get_writable_timeseries_view(
            ("Surface Temperature",),
            ("World",),
            "K",
            self._run_start,
            self._timestep,
        ).set_series(temperature)

        for name, info in _emissions_units_index_fair.items():
            if name == ("time",):
                continue
            results.parameters.get_writable_timeseries_view(
                ("Emissions",) + name,
                ("World",),
                info["unit"],
                self._run_start,
                self._timestep,
            ).set_series(self._drivers["emissions"][:, info["index"]])

        for name, info in _concentrations_units_index_fair.items():
            results.parameters.get_writable_timeseries_view(
                ("Atmospheric Concentrations",) + name,
                ("World",),
                info["unit"],
                self._run_start,
                self._timestep,
            ).set_series(concs[:, info["index"]])

        for name, info in _radiative_forcing_units_index_fair.items():
            results.parameters.get_writable_timeseries_view(
                ("Radiative Forcing",) + name,
                ("World",),
                info["unit"],
                self._run_start,
                self._timestep,
            ).set_series(forcing[:, info["index"]])

        for name, info in _parameters_fair.items():
            try:
                value = self._config[name]
            except KeyError:
                # this is where Sven's retrieval of default parameters idea comes
                # in handy, although it may be slow...
                signature = inspect.signature(self.model)
                # this will explode when we start including array parameters...
                if name in ("ecs", "tcr"):
                    tcrecs_def = signature.parameters["tcrecs"].default
                    if name == "ecs":
                        value = tcrecs_def[1]
                    else:
                        value = tcrecs_def[1]
                else:
                    value = signature.parameters[name].default

            results.parameters.get_writable_scalar_view(
                (name,), ("World"), info["unit"]
            ).set(value)

        for openscm_name, fair_name in _map.items():
            try:
                value = self._config[fair_name]
            except KeyError:
                # this is where Sven's retrieval of default parameters idea comes
                # in handy, although it may be slow...
                signature = inspect.signature(self.model)
                # this will explode when we start including array parameters...
                value = signature.parameters[fair_name].default

            results.parameters.get_writable_scalar_view(
                (openscm_name,), ("World"), "W / m^2"
            ).set(value)

        return results

    def step(self) -> None:
        raise NotImplementedError("Although should be doable...")

    def shutdown(self) -> None:
        pass

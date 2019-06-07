"""
Adapter for the simple climate model first presented in Petschel-Held Climatic Change 1999.
"""
import warnings

import numpy as np

from ..core.parameters import ParameterType
from ..core.time import create_time_points
from ..core.units import _unit_registry
from ..errors import ParameterEmptyError
from ..models import PH99Model
from . import Adapter


class PH99(Adapter):
    """
    Adapter for the simple climate model first presented in Petschel-Held Climatic Change 1999.

    This one box model projects global-mean |CO2| concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of |CO2| alone.

    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """

    _hc_per_m2_approx = 1.34 * 10 ** 9 * _unit_registry("J / kelvin / m^2")
    """Approximate heat capacity per unit area (used to estimate rf2xco2)"""

    _base_time = np.datetime64("1750-01-01")
    """Base time. PH99 has no concept of datetimes so we make it up here"""

    @property
    def _ecs(self):
        return self._parameters.scalar("Equilibrium Climate Sensitivity", "delta_degC").value * _unit_registry("delta_degC")


    def _initialize_model(self) -> None:
        """
        Initialize the model.
        """
        self.model = PH99Model()  # pylint: disable=attribute-defined-outside-init

        for att in dir(self.model):
            if not att.startswith(("_", "emissions_idx")):
                value = getattr(self.model, att)
                if callable(value):
                    continue

                name = self._get_openscm_name(att)
                if isinstance(value.magnitude, np.ndarray):
                    continue
                self._set_default_value(name, value)

        self._initialize_openscm_standard_parameters()

    def _set_default_value(self, name, value):
        if name == ("PH99", "timestep"):
            units = "s"
            mag = int(value.to(units).magnitude)
            self._parameters.scalar(name, "s").value = int(mag)
            if int(self.model.timestep.magnitude) != int(mag):
                warnings.warn("Rounding model timestep to nearest integer")
                self.model.timestep = mag * _unit_registry("s")  # ensure we use integer steps to be OpenSCM compatible
            return

        p = self._parameters.scalar(name, str(value.units))
        if p.empty:
            p.value = value.magnitude

    def _initialize_openscm_standard_parameters(self):
        start_time = self._parameters.generic("Start Time")
        if start_time.empty:
            start_time.value = (
                self._base_time
                + np.timedelta64(int(self.model.time_start.to("s").magnitude), "s")
            )

        ecs = self._parameters.scalar(
            "Equilibrium Climate Sensitivity",
            "delta_degC"
        )
        if ecs.empty:
            ecs.value = (self.model.mu * np.log(2) / self.model.alpha).to("delta_degC").magnitude
        else:
            self._set_model_parameter("Equilibrium Climate Sensitivity", ecs.value * _unit_registry("delta_degC"))

        rf2xco2 = self._parameters.scalar(
            "Radiative Forcing 2xCO2",
            "W/m^2"
        )
        if rf2xco2.empty:
            rf2xco2.value = (self.model.mu * self._hc_per_m2_approx).to("W/m^2").magnitude
        else:
            self._set_model_parameter("Radiative Forcing 2xCO2", rf2xco2.value * _unit_registry("W/m^2"))


    def _initialize_model_input(self) -> None:
        pass

    def _initialize_run_parameters(self) -> None:
        # TODO: make this easier
        for (
            key,
            value,
        ) in (
            self._parameters._root._parameters.items()  # pylint: disable=protected-access
        ):
            try:
                value.data  # pylint: disable=protected-access
            except AttributeError:
                continue
            if key == "Start Time":
                self.model.time_start = (value.data - self._base_time).item().total_seconds() * _unit_registry("s")
                continue
            if key == "Stop Time":
                continue  # set below

            self._set_model_parameter(key, value.data * _unit_registry(value.unit))

        start_time = self._parameters.generic("Start Time").value
        stop_time = self._parameters.generic("Stop Time").value
        timestep_count = int(
            (stop_time - start_time).item().total_seconds()
            // int(self.model.timestep.to("s").magnitude)
            + 1
        )

        time_points = create_time_points(
            start_time,
            np.timedelta64(int(self.model.timestep.to("s").magnitude), "s"),
            timestep_count,
            timeseries_type="average",
        )

        emms_units = self.model.emissions.units
        try:
            self.model.emissions = (
                self._parameters.timeseries(
                    ("Emissions", "CO2"),
                    str(emms_units),
                    time_points,
                    region=("World",),
                    timeseries_type="average",
                    interpolation="linear",
                ).values
                * emms_units
            )
        except ParameterEmptyError:
            raise ParameterEmptyError(
                "PH99 requires ('Emissions', 'CO2') in order to run"
            )

    def _set_model_parameter(self, para_name, value):
        if para_name == "Equilibrium Climate Sensitivity":
            # TODO: decide how to handle contradiction in a more sophisticated way
            alpha_val = getattr(self.model, "mu") * np.log(2) / value
            warnings.warn(
                "Updating Equilibrium Climate Sensitivity also updates alpha"
            )
            self._update_model_parameter_and_parameterset("alpha", alpha_val)

            return

        if para_name == "Radiative Forcing 2xCO2":
            # TODO: decide how to handle contradiction in a more sophisticated way
            mu_val = (value / self._hc_per_m2_approx).to(self.model.mu.units)
            warnings.warn("Updating Radiative Forcing 2xCO2 also updates mu")
            self._update_model_parameter_and_parameterset("mu", mu_val)

            # reset alpha too as it depends on mu
            alpha_val = getattr(self.model, "mu") * np.log(2) / self._ecs
            warnings.warn("Updating rf2xco2 also updates alpha")
            self._update_model_parameter_and_parameterset("alpha", alpha_val)

            return

        try:
            # TODO: make this easier
            value = value
            setattr(
                self.model, para_name, value.to(getattr(self.model, para_name).units)
            )
        except AttributeError:
            # TODO: make this more controlled elsewhere
            warnings.warn("Not using {}".format(para_name))

    def _update_model_parameter_and_parameterset(self, para_name, value):
        setattr(self.model, para_name, value.to(getattr(self.model, para_name).units))
        self._parameters.scalar(
            ("PH99", para_name), str(value.units), region=("World",)
        ).value = value.magnitude

    def _reset(self) -> None:
        # reset to whatever is in the views of self
        # probably requires a parameters and a timeseries list/method to get it...
        pass

    def _shutdown(self) -> None:
        pass

    def _get_openscm_name(self, name):  # pylint: disable=no-self-use
        mappings = {
            "concentrations": ("Atmospheric Concentrations", "CO2"),
            "cumulative_emissions": ("Cumulative Emissions", "CO2"),
            "emissions": ("Emissions", "CO2"),
            "temperatures": ("Surface Temperature"),
        }
        try:
            return mappings[name]
        except KeyError:
            return ("PH99", name)

    def _run(self) -> None:
        self.model.initialise_timeseries()
        self.model.run()

        # self._output = ParameterSet()

        for att in dir(self.model):
            # all time parameters captured in parameterset output
            if not att.startswith(("_", "time", "emissions_idx")):
                value = getattr(self.model, att)
                if callable(value):
                    continue

                name = self._get_openscm_name(att)
                magnitude = value.magnitude
                if isinstance(magnitude, np.ndarray):
                    if name == ("Surface Temperature"):
                        # this is where reference period etc. is important
                        value = value - self.model.t1
                        magnitude = value.magnitude

                    if name in [("Emissions", "CO2")]:
                        ptype = ParameterType.AVERAGE_TIMESERIES
                    else:
                        ptype = ParameterType.POINT_TIMESERIES

                    time_points = create_time_points(
                        self._parameters.generic("Start Time").value,
                        np.timedelta64(int(self.model.timestep.to("s").magnitude), "s"),
                        len(magnitude),
                        timeseries_type=ptype,
                    )
                    self._output.timeseries(
                        name,
                        str(value.units),
                        time_points,
                        region=("World",),
                        timeseries_type=ptype,
                    ).values = magnitude
                else:
                    self._output.scalar(
                        name, str(value.units), region=("World")
                    ).value = magnitude

        ecs = (self.model.mu * np.log(2) / self.model.alpha).to("K")
        self._output.scalar(
            ("ecs",), str(ecs.units), region=("World",)
        ).value = ecs.magnitude

        rf2xco2 = self.model.mu * self._hc_per_m2_approx
        self._output.scalar(
            ("rf2xco2",), str(rf2xco2.units), region=("World",)
        ).value = rf2xco2.magnitude

    def _step(self) -> None:
        self.model.initialise_timeseries()
        self.model.step()
        self._current_time = self._parameters.generic(
            "Start Time"
        ).value + np.timedelta64(int(self.model.time_current.to("s").magnitude), "s")
        # TODO: update output

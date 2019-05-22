"""
Adapter for the simple climate model first presented in Petschel-Held Climatic Change 1999.
"""
import warnings

import numpy as np

from ..adapter import Adapter
from ..errors import ParameterEmptyError
from ..models import PH99Model
from ..parameters import ParameterType
from ..timeseries_converter import InterpolationType, create_time_points
from ..units import _unit_registry


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
    _ecs = None

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
                magnitude = value.magnitude
                if isinstance(magnitude, np.ndarray):
                    continue

                try:
                    self._parameters.get_scalar_view(
                        name, ("World",), str(value.units)
                    ).get()
                except ParameterEmptyError:
                    self._parameters.get_writable_scalar_view(
                        name, ("World",), str(value.units)
                    ).set(magnitude)

        self._ecs = self.model.mu * np.log(2) / self.model.alpha

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
                value._data  # pylint: disable=protected-access
            except AttributeError:
                continue
            self._set_model_parameter(key, value)

        timestep_count = (self._stop_time - self._start_time) // int(
            self.model.timestep.to("s").magnitude
        ) + 1

        time_points = create_time_points(
            self._start_time,
            self.model.timestep.to("s").magnitude,
            timestep_count,
            ParameterType.AVERAGE_TIMESERIES,
        )

        emms_units = self.model.emissions.units
        emms_view = self._parameters.get_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            str(emms_units),
            time_points,
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
        )
        if emms_view.is_empty:
            raise ParameterEmptyError(
                "PH99 requires ('Emissions', 'CO2') in order to run"
            )

        self.model.emissions = emms_view.get() * emms_units

    def _set_model_parameter(self, para_name, value):
        try:
            # TODO: make this easier
            modval = _unit_registry.Quantity(
                value._data, value.info.unit  # pylint: disable=protected-access
            )
            setattr(
                self.model, para_name, modval.to(getattr(self.model, para_name).units)
            )
        except AttributeError:
            if para_name == "ecs":
                # I'm not sure this is correct, check properly when making proper PR
                self._ecs = modval

                alpha_val = getattr(self.model, "mu") * np.log(2) / modval
                self.model.alpha = alpha_val.to(self.model.alpha.units)
                self._parameters.get_scalar_view(
                    ("alpha",), ("World",), str(self.model.alpha.units)
                ).get()
                # TODO: decide how to handle contradiction in a more sophisticated way
                warnings.warn("Updating ecs also updates alpha")
                self._parameters.get_writable_scalar_view(
                    ("alpha",), ("World",), str(self.model.alpha.units)
                ).set(alpha_val.magnitude)

                return

            if para_name == "rf2xco2":
                # I'm not sure this is correct, check properly when making proper PR
                mu_val = (modval / self._hc_per_m2_approx).to(self.model.mu.units)
                self.model.mu = mu_val
                self._parameters.get_scalar_view(
                    ("mu",), ("World",), str(self.model.mu.units)
                ).get()
                # TODO: decide how to handle contradiction in a more sophisticated way
                warnings.warn("Updating rf2xco2 also updates mu")
                self._parameters.get_writable_scalar_view(
                    ("mu",), ("World",), str(self.model.mu.units)
                ).set(mu_val.magnitude)

                # reset alpha too as it depends on mu
                alpha_val = getattr(self.model, "mu") * np.log(2) / self._ecs
                self.model.alpha = alpha_val.to(self.model.alpha.units)
                self._parameters.get_scalar_view(
                    ("alpha",), ("World",), str(self.model.alpha.units)
                ).get()
                # TODO: decide how to handle contradiction in a more sophisticated way
                warnings.warn("Updating rf2xco2 also updates alpha")
                self._parameters.get_writable_scalar_view(
                    ("alpha",), ("World",), str(self.model.alpha.units)
                ).set(alpha_val.magnitude)

                return

            if para_name == "start_time":
                self.model.time_start = modval.to(self.model.time_start.units)
                return

            # TODO: make this more controlled elsewhere
            warnings.warn("Not using {}".format(para_name))

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
            return (name,)

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
                        self.model.time_start.to("s").magnitude,
                        self.model.timestep.to("s").magnitude,
                        len(magnitude),
                        ptype,
                    )
                    self._output.get_writable_timeseries_view(
                        name, ("World",), str(value.units), time_points, ptype
                    ).set(magnitude)
                else:
                    self._output.get_writable_scalar_view(
                        name, ("World"), str(value.units)
                    ).set(magnitude)

        ecs = (self.model.mu * np.log(2) / self.model.alpha).to("K")
        self._output.get_writable_scalar_view(("ecs",), ("World",), str(ecs.units)).set(
            ecs.magnitude
        )

        rf2xco2 = self.model.mu * self._hc_per_m2_approx
        self._output.get_writable_scalar_view(
            ("rf2xco2",), ("World",), str(rf2xco2.units)
        ).set(rf2xco2.magnitude)

    def _step(self) -> None:
        self.model.initialise_timeseries()
        self.model.step()
        self._current_time = self.model.time_current.to("s").magnitude
        # TODO: update output

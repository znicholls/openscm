import datetime as dt
import re

import numpy as np
import pytest
from base import _AdapterTester
from conftest import assert_pint_equal

from openscm.adapters.ph99 import PH99
from openscm.core import ParameterSet
from openscm.core.parameters import ParameterType

# from openscm.core.timeseries_converter import InterpolationType, create_time_points
from openscm.core.time import InterpolationType, create_time_points
from openscm.core.units import _unit_registry
from openscm.errors import ParameterEmptyError

# from openscm.utils import convert_datetime_to_openscm_time


class TestPH99Adapter(_AdapterTester):
    tadapter = PH99

    def test_initialize_model_input_ph99_specific(self):
        in_parameters = ParameterSet()
        out_parameters = ParameterSet()
        tadapter = self.tadapter(in_parameters, out_parameters)

        with pytest.raises(AttributeError):
            tadapter.model
        tadapter.initialize_model_input()
        assert tadapter.model

        assert (
            in_parameters.scalar(
                ("PH99", "b"), "ppm / (GtC * yr)", region=("World",)
            ).value
            == 1.51 * 10 ** -3
        )
        assert (
            in_parameters.scalar(("PH99", "c1"), "ppm", region=("World",)).value == 290
        )
        assert (
            in_parameters.scalar(("PH99", "t1"), "K", region=("World",)).value == 287.75
        )
        assert in_parameters.generic(
            ("Start Time",), region=("World",)
        ).value == np.datetime64("1750-01-01")

    def test_openscm_standard_parameters_take_priority(self, test_drivers):
        expected = test_drivers["setters"]
        in_parameters = test_drivers["ParameterSet"]

        out_parameters = ParameterSet()
        tadapter = self.tadapter(in_parameters, out_parameters)

        rf2xco2 = 3.5
        in_parameters.scalar(
            ("Radiative Forcing 2xCO2",), "W/m^2", region=("World",)
        ).value = rf2xco2

        mu = 8.9 * 10 ** -2
        in_parameters.scalar(("PH99", "mu"), "degC/yr", region=("World",)).value = mu

        alpha = 1.9 * 10 ** -2
        in_parameters.scalar(("PH99", "alpha"), "1/yr", region=("World",)).value = alpha

        tadapter.initialize_run_parameters()

        assert_pint_equal(
            tadapter.model.alpha,
            tadapter.model.mu * np.log(2) / expected["Equilibrium Climate Sensitivity"],
        )

        expected_mu = (
            _unit_registry.Quantity(rf2xco2, "W/m^2") / tadapter._hc_per_m2_approx
        )
        assert_pint_equal(tadapter.model.mu, expected_mu)
        np.testing.assert_allclose(
            in_parameters.scalar(("PH99", "mu"), "degC/yr", region=("World",)).value,
            expected_mu.to("degC/yr"),
        )

        # make sure tadapter.model.mu isn't given by value passed into ParameterSet
        # earlier i.e. openscm parameter takes priority
        with pytest.raises(AssertionError):
            assert_pint_equal(tadapter.model.mu, _unit_registry.Quantity(mu, "degC/yr"))

        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                in_parameters.scalar(
                    ("PH99", "mu"), "degC/yr", region=("World",)
                ).value,
                mu,
            )

        np.testing.assert_allclose(
            in_parameters.scalar(
                ("Radiative Forcing 2xCO2",), "W/m^2", region=("World",)
            ).value,
            rf2xco2,
        )

        in_parameters.scalar(
            ("Radiative Forcing 2xCO2",), "W/m^2", region=("World",)
        ).value = (2 * rf2xco2)
        tadapter.initialize_run_parameters()

        assert_pint_equal(
            tadapter.model.alpha,
            tadapter.model.mu * np.log(2) / expected["Equilibrium Climate Sensitivity"],
        )
        assert_pint_equal(
            tadapter.model.mu,
            _unit_registry.Quantity(2 * rf2xco2, "W/m^2") / tadapter._hc_per_m2_approx,
        )
        np.testing.assert_allclose(
            in_parameters.scalar(
                ("Radiative Forcing 2xCO2",), "W/m^2", region=("World",)
            ).value,
            2 * rf2xco2,
        )

        np.testing.assert_allclose(
            tadapter._parameters.scalar(
                ("Radiative Forcing 2xCO2",), "W/m^2", region=("World",)
            ).value,
            2 * rf2xco2,
        )

    def test_initialize_run_parameters(self, test_adapter, test_run_parameters):
        test_adapter._parameters.generic(
            "Start Time"
        ).value = test_run_parameters.start_time
        test_adapter._parameters.generic(
            "Stop Time"
        ).value = test_run_parameters.stop_time
        tp = create_time_points(  # TODO: replace by simpler function
            test_run_parameters.start_time,
            np.timedelta64(365 * 50, "D"),
            3,
            timeseries_type="average",
        )
        test_adapter._parameters.timeseries(
            ("Emissions", "CO2"), "GtC/day", tp, timeseries_type="average"
        ).values = np.array([10, 20, 21])
        super().test_initialize_run_parameters(test_adapter, test_run_parameters)

    def test_initialize_run_parameters_ph99_specific(self, test_drivers):
        expected = test_drivers["setters"]
        in_parameters = test_drivers["ParameterSet"]

        out_parameters = ParameterSet()
        tadapter = self.tadapter(in_parameters, out_parameters)

        tc1 = 3.8
        in_parameters.scalar(("PH99", "c1"), "ppb", region=("World",)).value = (
            tc1 * 1000
        )

        tadapter.initialize_run_parameters()
        np.testing.assert_allclose(
            tadapter._parameters.scalar(("PH99", "c1"), "ppm", region=("World",)).value,
            tc1,
        )

        timestep = tadapter.model.timestep.to("s").magnitude
        assert (
            timestep
            == in_parameters.scalar(("PH99", "timestep"), "s", region="World").value
        )
        timestep_count = (
            int(
                (expected["stop_time"] - expected["start_time"]).item().total_seconds()
                // timestep
            )
            + 1
        )
        time_points = create_time_points(
            expected["start_time"],
            np.timedelta64(int(timestep), "s"),
            timestep_count,
            timeseries_type="average",
        )
        expected_emms = in_parameters.timeseries(
            ("Emissions", "CO2"),
            str(tadapter.model.emissions.units),
            time_points,
            timeseries_type="average",
            interpolation="linear",
        ).values
        np.testing.assert_allclose(tadapter.model.emissions, expected_emms)

    def prepare_run_input(self, test_adapter, start_time, stop_time):
        """
        Overload this in your adapter test if you need to set required input parameters.
        This method is called directly before ``test_adapter.initialize_model_input``
        during tests.
        """
        test_adapter._parameters.generic("Start Time").value = start_time
        test_adapter._parameters.generic("Stop Time").value = stop_time
        timestep = np.timedelta64(30, "D")
        test_adapter._parameters.scalar(
            ("PH99", "timestep"), "s"
        ).value = timestep.item().total_seconds()

        npoints = (stop_time - start_time) // timestep + 1
        time_points_for_averages = create_time_points(
            start_time,
            stop_time - start_time,
            npoints,
            ParameterType.AVERAGE_TIMESERIES,
        )
        test_adapter._parameters.timeseries(
            ("Emissions", "CO2"),
            "GtCO2/a",
            time_points_for_averages,
            timeseries_type="average",
        ).values = np.zeros(npoints)

    def test_run_ph99_specific(self, test_drivers):
        expected = test_drivers["setters"]
        in_parameters = test_drivers["ParameterSet"]
        out_parameters = ParameterSet()
        tadapter = self.tadapter(in_parameters, out_parameters)

        tadapter.initialize_model_input()
        tadapter.initialize_run_parameters()
        tadapter.reset()
        tadapter.run()

        timestep = tadapter.model.timestep.to("s").magnitude
        assert timestep == in_parameters.scalar(("PH99", "timestep"), "s").value
        timestep_count = (
            int(
                (expected["stop_time"] - expected["start_time"]).item().total_seconds()
                // timestep
            )
            + 1
        )
        time_points = create_time_points(
            expected["start_time"],
            np.timedelta64(int(timestep), "s"),
            timestep_count,
            ParameterType.AVERAGE_TIMESERIES,
        )
        expected_emms = in_parameters.timeseries(
            ("Emissions", "CO2"),
            str(tadapter.model.emissions.units),
            time_points,
            timeseries_type="average",
            interpolation="linear",
        ).values

        resulting_emms = out_parameters.timeseries(
            ("Emissions", "CO2"),
            str(tadapter.model.emissions.units),
            time_points,
            timeseries_type="average",
            interpolation="linear",
        ).values

        np.testing.assert_allclose(
            expected_emms, resulting_emms, rtol=1e-10, atol=max(expected_emms) * 1e-6
        )

        # regression test
        temp_2017_2018 = tadapter._output.timeseries(
            ("Surface Temperature"),
            "K",
            np.array(
                [
                    np.datetime64("2017-01-01").astype("datetime64[s]").astype(float),
                    np.datetime64("2018-01-01").astype("datetime64[s]").astype(float),
                ]
            ),
            region="World",
            timeseries_type="point",
        ).values
        np.testing.assert_allclose(
            temp_2017_2018,
            np.array([0.5240462684400263, 0.5296034389009026]),
            rtol=1e-5,
        )

    def test_run_no_emissions_error(self, test_adapter):
        test_adapter._parameters.generic("Start Time").value = np.datetime64(
            "2010-01-01"
        )
        test_adapter._parameters.generic("Stop Time").value = np.datetime64(
            "2013-01-01"
        )
        test_adapter.initialize_model_input()
        error_msg = re.escape("PH99 requires ('Emissions', 'CO2') in order to run")
        with pytest.raises(ParameterEmptyError, match=error_msg):
            test_adapter.initialize_run_parameters()

    def test_openscm_standard_parameters_handling(self):
        parameters = ParameterSet()

        start_t = np.datetime64("1850-01-01")
        parameters.generic("Start Time").value = start_t

        stop_t = np.datetime64("2100-01-01")
        parameters.generic("Stop Time").value = stop_t

        ecs_magnitude = 3.12
        parameters.scalar(
            "Equilibrium Climate Sensitivity", "delta_degC"
        ).value = ecs_magnitude
        parameters.scalar(("PH99", "alpha"), "1/yr").value = (
            3.9 * 10 ** -2
        )  # ensure openscm standard parameters take precedence

        rf2xco2_magnitude = 4.012
        parameters.scalar(
            "Radiative Forcing 2xCO2", "W / m^2"
        ).value = rf2xco2_magnitude

        output_parameters = ParameterSet()

        test_adapter = self.tadapter(parameters, output_parameters)

        self.prepare_run_input(
            test_adapter,
            parameters.generic("Start Time").value,
            parameters.generic("Stop Time").value,
        )
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters()
        test_adapter.reset()
        test_adapter.run()

        assert test_adapter._parameters.generic("Start Time").value == start_t
        assert test_adapter._parameters.generic("Stop Time").value == stop_t
        assert (
            test_adapter._parameters.scalar(
                "Equilibrium Climate Sensitivity", "delta_degC"
            ).value
            == ecs_magnitude
        )
        assert (
            test_adapter._parameters.scalar("Radiative Forcing 2xCO2", "W/m^2").value
            == rf2xco2_magnitude
        )

        # do we want adapters to push all parameter values to output too? If yes, uncomment this
        # assert output_parameters.generic("Start Time").value == np.datetime64("1850-01-01")
        # assert output_parameters.generic("Stop Time").value == np.datetime64("2100-01-01")
        # assert output_parameters.scalar("Equilibrium Climate Sensitivity", "delta_degC").value == ecs_magnitude

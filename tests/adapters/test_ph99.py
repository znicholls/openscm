import datetime as dt
import re

import numpy as np
import pytest
from base import _AdapterTester
from conftest import assert_pint_equal

from openscm.adapters.ph99 import PH99
from openscm.core import ParameterSet
from openscm.errors import ParameterEmptyError
from openscm.core.parameters import ParameterType
# from openscm.core.timeseries_converter import InterpolationType, create_time_points
from openscm.core.time import InterpolationType, create_time_points
from openscm.core.units import _unit_registry
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
            in_parameters.get_scalar_view(
                ("PH99", "b"), ("World",), "ppm / (GtC * yr)"
            ).get()
            == 1.51 * 10 ** -3
        )
        assert (
            in_parameters.get_scalar_view(("PH99", "c1"), ("World",), "ppm").get()
            == 290
        )
        assert (
            in_parameters.get_scalar_view(("PH99", "t1"), ("World",), "K").get()
            == 287.75
        )
        assert in_parameters.get_scalar_view(
            ("start_time",), ("World",), "s"
        ).get() == convert_datetime_to_openscm_time(dt.datetime(1750, 1, 1))
        assert in_parameters.get_scalar_view(
            ("stop_time",), ("World",), "s"
        ).get() == convert_datetime_to_openscm_time(dt.datetime(2500, 1, 1))

    def test_openscm_standard_parameters_take_priority(self, test_drivers):
        expected = test_drivers["setters"]
        in_parameters = test_drivers["ParameterSet"]

        out_parameters = ParameterSet()
        tadapter = self.tadapter(in_parameters, out_parameters)

        rf2xco2 = 3.5
        in_parameters.get_writable_scalar_view(("rf2xco2",), ("World",), "W/m^2").set(
            rf2xco2
        )

        mu = 8.9 * 10 ** -2
        in_parameters.get_writable_scalar_view(
            ("PH99", "mu"), ("World",), "degC/yr"
        ).set(mu)

        alpha = 1.9 * 10 ** -2
        in_parameters.get_writable_scalar_view(
            ("PH99", "alpha"), ("World",), "1/yr"
        ).set(alpha)

        tadapter.initialize_run_parameters()

        assert_pint_equal(
            tadapter.model.alpha, tadapter.model.mu * np.log(2) / expected["ecs"]
        )

        expected_mu = (
            _unit_registry.Quantity(rf2xco2, "W/m^2") / tadapter._hc_per_m2_approx
        )
        assert_pint_equal(tadapter.model.mu, expected_mu)
        np.testing.assert_allclose(
            in_parameters.get_scalar_view(("PH99", "mu"), ("World",), "degC/yr").get(),
            expected_mu.to("degC/yr"),
        )

        # make sure tadapter.model.mu isn't given by value passed into ParameterSet
        # earlier i.e. openscm parameter takes priority
        with pytest.raises(AssertionError):
            assert_pint_equal(tadapter.model.mu, _unit_registry.Quantity(mu, "degC/yr"))

        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                in_parameters.get_scalar_view(
                    ("PH99", "mu"), ("World",), "degC/yr"
                ).get(),
                mu,
            )

        np.testing.assert_allclose(
            in_parameters.get_scalar_view(("rf2xco2",), ("World",), "W/m^2").get(),
            rf2xco2,
        )

        in_parameters.get_writable_scalar_view(("rf2xco2",), ("World",), "W/m^2").set(
            2 * rf2xco2
        )
        tadapter.initialize_run_parameters()

        assert_pint_equal(
            tadapter.model.alpha, tadapter.model.mu * np.log(2) / expected["ecs"]
        )
        assert_pint_equal(
            tadapter.model.mu,
            _unit_registry.Quantity(2 * rf2xco2, "W/m^2") / tadapter._hc_per_m2_approx,
        )
        np.testing.assert_allclose(
            in_parameters.get_scalar_view(("rf2xco2",), ("World",), "W/m^2").get(),
            2 * rf2xco2,
        )

        np.testing.assert_allclose(
            tadapter._parameters.get_scalar_view(
                ("rf2xco2",), ("World",), "W/m^2"
            ).get(),
            2 * rf2xco2,
        )

    def test_initialize_run_parameters_ph99_specific(self, test_drivers):
        expected = test_drivers["setters"]
        in_parameters = test_drivers["ParameterSet"]

        out_parameters = ParameterSet()
        tadapter = self.tadapter(in_parameters, out_parameters)

        tc1 = 3.8
        in_parameters.get_writable_scalar_view(("PH99", "c1"), ("World",), "ppb").set(
            tc1 * 1000
        )
        tadapter.initialize_run_parameters()
        np.testing.assert_allclose(
            tadapter._parameters.get_scalar_view(
                ("PH99", "c1"), ("World",), "ppm"
            ).get(),
            tc1,
        )

        timestep = tadapter.model.timestep.to("s").magnitude
        assert (
            timestep
            == in_parameters.get_scalar_view(("PH99", "timestep"), "World", "s").get()
        )
        timestep_count = (
            expected["stop_time"] - expected["start_time"]
        ) // timestep + 1
        time_points = create_time_points(
            expected["start_time"],
            timestep,
            timestep_count,
            ParameterType.AVERAGE_TIMESERIES,
        )
        expected_emms = in_parameters.get_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            str(tadapter.model.emissions.units),
            time_points,
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
        ).get()
        np.testing.assert_allclose(tadapter.model.emissions, expected_emms)

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
        assert (
            timestep
            == in_parameters.get_scalar_view(("PH99", "timestep"), "World", "s").get()
        )
        timestep_count = (
            expected["stop_time"] - expected["start_time"]
        ) // timestep + 1
        time_points = create_time_points(
            expected["start_time"],
            timestep,
            timestep_count,
            ParameterType.AVERAGE_TIMESERIES,
        )
        expected_emms = in_parameters.get_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            str(tadapter.model.emissions.units),
            time_points,
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
        ).get()

        resulting_emms = out_parameters.get_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            str(tadapter.model.emissions.units),
            time_points,
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
        ).get()

        np.testing.assert_allclose(
            expected_emms, resulting_emms, rtol=1e-10, atol=max(expected_emms) * 1e-6
        )

        # regression test
        temp_2017_2018 = tadapter._output.get_timeseries_view(
            ("Surface Temperature"),
            "World",
            "K",
            np.array(
                [
                    convert_datetime_to_openscm_time(dt.datetime(2017, 1, 1)),
                    convert_datetime_to_openscm_time(dt.datetime(2018, 1, 1)),
                ]
            ),
            ParameterType.POINT_TIMESERIES,
        ).get()
        np.testing.assert_allclose(
            temp_2017_2018, np.array([-0.00142103, -0.0024701]), rtol=1e-5
        )

    def test_run_no_emissions_error(self, test_adapter):
        test_adapter.initialize_model_input()
        error_msg = re.escape("PH99 requires ('Emissions', 'CO2') in order to run")
        with pytest.raises(ParameterEmptyError, match=error_msg):
            test_adapter.initialize_run_parameters()

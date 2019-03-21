import datetime as dt
from copy import deepcopy

import numpy as np
import pytest
from base import _AdapterTester

from openscm.adapters.ph99 import PH99
from openscm.core import ParameterSet
from openscm.parameters import ParameterType
from openscm.timeseries_converter import InterpolationType, create_time_points
from openscm.utils import convert_datetime_to_openscm_time


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
            in_parameters.get_scalar_view(("b",), ("World",), "ppm / (GtC * yr)").get()
            == 1.51 * 10**-3
        )
        assert (
            in_parameters.get_scalar_view(("c1",), ("World",), "ppm").get()
            == 290
        )
        assert (
            in_parameters.get_scalar_view(("t1",), ("World",), "K").get()
            == 287.75
        )
        assert (
            in_parameters.get_scalar_view(("start_time",), ("World",), "s").get()
            == convert_datetime_to_openscm_time(dt.datetime(1750, 1, 1))
        )
        assert (
            in_parameters.get_scalar_view(("stop_time",), ("World",), "s").get()
            == convert_datetime_to_openscm_time(dt.datetime(2500, 1, 1))
        )


    def test_initialize_run_parameters_ph99_specific(self, test_drivers):
        expected = test_drivers["setters"]
        in_parameters = test_drivers["ParameterSet"]

        out_parameters = ParameterSet()
        tadapter = self.tadapter(in_parameters, out_parameters)

        tc1 = 3.8
        in_parameters.get_writable_scalar_view(("c1",), ("World",), "ppb").set(
            tc1 * 1000
        )
        tadapter.initialize_run_parameters()
        np.testing.assert_allclose(
            tadapter._parameters.get_scalar_view(("c1",), ("World",), "ppm").get(),
            tc1
        )

        timestep = tadapter.model.timestep.to("s").magnitude
        assert timestep == in_parameters.get_scalar_view("timestep", "World", "s").get()
        timestep_count = (
            (expected["stop_time"] - expected["start_time"])
            // timestep
            + 1
        )
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
        np.testing.assert_allclose(
            tadapter.model.emissions,
            expected_emms
        )

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
        assert timestep == in_parameters.get_scalar_view("timestep", "World", "s").get()
        timestep_count = (
            (expected["stop_time"] - expected["start_time"])
            // timestep
            + 1
        )
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
            expected_emms,
            resulting_emms,
            rtol=1e-10,
            atol=max(expected_emms)*1e-6,
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
            temp_2017_2018, np.array([-0.0008942013640017765, -0.00202055345682164])
        )

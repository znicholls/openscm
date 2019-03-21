from openscm.parameters import ParameterType

class _AdapterTester:
    """
    Base class for adapter testing.

    At minimum, a new adapter should define a subclass of this class called,
    ``AdapterXTester`` which has ``tadapter`` set to the adapter to be tested. This
    ensures that the new adapter is subject to all of OpenSCM's minimum requirements
    whilst giving authors the ability to tweak the tests as necessary for their specific
    adapter.
    """

    tadapter = None
    """
    Adapter to test
    """

    def test_initialize(self, test_adapter):
        test_adapter._initialize_model()

        # This comes from how test_adapter is initialised, we might want to get
        # rid of this in future as it will interfere with defaults.
        assert test_adapter._parameters.get_scalar_view(
            ("rf2xco2",),
            ("World",),
            "W / m^2",
        ).get() == 4

        # Would it make sense for us to also pre-set the outputs with empty arrays?
        # Then we could use these outputs to help us with separating model config
        # from model timeseries. It may require a re-name, but that would also make
        # it possible for the `initialize_model_input` and `initialize_run_parameters`
        # methods to know which parameters they're meant to look at for the
        # initialisation.
        assert test_adapter._output.get_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            "GtCO2/yr",
            [0, 1, 2],
            ParameterType.AVERAGE_TIMESERIES,
        ).is_empty

        assert test_adapter._output.get_timeseries_view(
            ("model specific para",),
            ("model region",),
            "model unit",
            [0, 1],
            ParameterType.POINT_TIMESERIES,
        ).is_empty

        # We could turn this into an abstract method so that implementers are forced
        # to do an assertion like the below.
        assert test_adapter._parameters.get_scalar_view(
            ("model specific para",),
            ("model region",),
            "model unit",
        ).get() == expected_default_value

    def test_shutdown(self, test_adapter):
        """
        Test the adapter can be shutdown.

        Extra tests can be adapted depending on what the adapter should actually
        do on shutdown.
        """
        del test_adapter

    def test_initialize_model_input(self, test_adapter):
        """
        Test that initalizing model input does as intended.
        """
        set_values = [0.9, 1.1]
        test_adapter._output.get_writable_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            "GtC/yr",
            [-10, 0, 10],
            ParameterType.POINT_TIMESERIES,
        ).set(set_values)

        assert not test_adapter._initialized
        # start and end time should be set here as they're part of setting the
        # timeseries views of emissions/concentrations/other drivers
        test_adapter.initialize_model_input(start_time, end_time)
        assert test_adapter._initialized
        # TODO test for start time and end time being passed properly
        assert test_adapter._output.get_writable_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            "GtC/yr",
            [-10, 0, 10],
            ParameterType.POINT_TIMESERIES,
        ).get() == set_values

        # TODO test for missing but mandatory input
        # We could turn this into an abstract method so that implementers are forced
        # to do an assertion like the below.
        test_adapter._output.get_writable_timeseries_view(
            ("Emissions", "junk"),
            ("World",),
            "GtC/yr",
            [-10, 0, 10],
            ParameterType.POINT_TIMESERIES,
        ).set(set_values)
        # if you try to initialise and run with a timeseries that can't be used,
        # you get a warning
        assert raises_warning(test_adapter.initialize_model_input(
            start_time,
            end_time
        ))

        test_adapter._output.get_writable_timeseries_view(
            ("Atmospheric Concentrations", "CO2"),
            ("World",),
            "ppm",
            [298, 300, 310],
            ParameterType.POINT_TIMESERIES,
        ).set(set_values)
        # If you initialise with a different run mode, the emissions get reset
        # in order to avoid confusion
        test_adapter.initialize_model_input(
            start_time,
            end_time,
            run_mode="concentration driven"
        )
        np.testing.assert_allclose(test_adapter._output.get_writable_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            "GtC/yr",
            [-10, 0, 10],
            ParameterType.POINT_TIMESERIES,
        ).get(), 0)

    def test_initialize_initialize_model_input_non_model_parameter(self, test_adapter):
        test_adapter._output.get_writable_timeseries_view(
            ("Emissions", "exotic gas"),
            ("World",),
            "GtE/yr",
            [-10, 0, 10],
            ParameterType.POINT_TIMESERIES,
        ).set([1, 2, 3])

        assert not test_adapter._initialized
        # if you try to initialise and run with an input that can't be used,
        # you get a warning
        assert raises_warning(test_adapter.initialize_model_input(start_time, end_time)    )
        error_msg = "model cannot be "
        # TODO test that ("Emissions", "exotic gas") has not been used

    def test_initialize_run_parameters(self, test_adapter):
        """
        Test that initalizing run parameters does as intended.
        """
        test_value = 5
        assert not test_adapter._initialized
        test_adapter.get_writable_scalar_view(
            ("model_para",),
            ("model region",),
            "ppt"
        ).set(test_value)
        test_adapter.initialize_run_parameters()
        assert test_adapter._initialized
        # We could turn this into an abstract method so that implementers are forced
        # to do an assertion like the below.
        assert test_adapter.model.model_para == test_value
        # TODO: test that conversion is passed through correctly

    def test_initialize_run_parameters_non_model_parameter(self, test_adapter):
        tname = ("junk",)
        test_adapter._parameters.get_writable_scalar_view(tname, ("World",), "K").set(4)
        # if you try to initialise and run with a parameter that can't be used,
        # you get a warning
        assert raises_warning(test_adapter.initialize_run_parameters())
        # TODO test that "junk" has not been used

    def test_run(self, test_adapter):
        # we should be able to update one ParameterSet with another one
        # somehow, then we could just update the adapter's drivers with e.g.
        # the rcps
        test_adapter._output.update(openscm.rcp26)
        test_adapter.initialize_model_input(
            start_time, stop_time, run_mode="emissions driven"
        )
        # this would then set ecs and tcr from test_adapter._parameters
        test_adapter.initialize_run_parameters()
        test_adapter.reset()  # Does this make sense having just initialised?
        test_adapter.run()
        # We could turn this into an abstract method so that implementers are forced
        # to do an assertion like the below.
        assert test_adapter._parameters.get_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            "GtC/yr",
            [-10, 0, 10, 20],
            ParameterType.AVERAGE_TIMESERIES,
        ).get() == expected_values
        assert test_adapter._parameters.get_timeseries_view(
            ("Surface Temperature",),
            ("World",),
            "K",
            [-10, 0, 10],
            ParameterType.POINT_TIMESERIES,
        ).get() == expected_values

    def test_step(self, test_adapter, test_run_parameters):
        test_adapter.initialize_model_input(
            start_time, stop_time
        )
        test_adapter.initialize_run_parameters()
        test_adapter.reset()
        assert test_adapter._current_time == test_run_parameters.start_time
        try:
            test_adapter.step()
            new_time = test_adapter.step()
            assert new_time > test_run_parameters.start_time

            # We could turn this into an abstract method so that implementers are forced
            # to do an assertion like the below.
            assert test_adapter._parameters.get_timeseries_view(
                ("Emissions", "CO2"),
                ("World",),
                "GtC/yr",
                [-10, 0, 10],
                ParameterType.AVERAGE_TIMESERIES,
            ).get() == expected_values
            assert test_adapter._parameters.get_timeseries_view(
                ("Surface Temperature",),
                ("World",),
                "K",
                [-10,0 ],
                ParameterType.POINT_TIMESERIES,
            ).get() == expected_values

        except NotImplementedError:
            pass

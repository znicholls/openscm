from abc import ABCMeta, abstractmethod


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
        """
        Test the adapter is initiated as intended.

        Extra tests can be added for different adapters, depending on whether there
        should be any other behaviour as part of ``__init__`` calls.
        """
        test_adapter._initialize_model()

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

        Extra tests can be adapted depending on what the adapter should actually
        do when initialised.
        """
        assert not test_adapter._initialized
        test_adapter.initialize_model_input()
        assert test_adapter._initialized

        # TODO: add tests something like below
        # assert (
        #     test_adapter._parameters.get_scalar_view(("ecs",), ("World",), "K").get()
        #     == 3
        # )
        # assert (
        #     test_adapter._parameters.get_scalar_view(
        #         ("rf2xco2",), ("World",), "W / m^2"
        #     ).get()
        #     == 4
        # )

    # TODO: move this into issue about run tests
    # def test_initialize_model_input_non_model_parameter(
    #     self, test_adapter, test_drivers
    # ):
    #     tname = ("junk",)
    #     test_adapter._parameters.get_writable_scalar_view(tname, ("World",), "K").set(4)
    #     test_adapter.initialize_model_input()
    #     # TODO test that "junk" has not been used

    def test_initialize_run_parameters(self, test_adapter, test_run_parameters):
        """
        Test that initalizing run parameters does as intended.
        """
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        # TODO: uncomment these tests and check passing worked as intended
        # expected = test_drivers["setters"]
        # in_parameters = test_drivers["ParameterSet"]
        # start_time = 30
        # stop_time = 250 * 365 * 24 * 60 * 60
        # in_parameters.get_writable_scalar_view(
        #     ("start_time",), ("World",), "s"
        # ).set(start_time)
        # in_parameters.get_writable_scalar_view(
        #     ("stop_time",), ("World",), "s"
        # ).set(stop_time)

        # out_parameters = ParameterSet()
        # tadapter = self.tadapter(in_parameters, out_parameters)

        assert not test_adapter._initialized
        test_adapter.initialize_run_parameters()
        assert test_adapter._initialized

        # TODO: uncomment these tests and check passing worked as intended
        # assert tadapter._start_time == start_time
        # assert tadapter._stop_time == stop_time

        # assert (
        #     in_parameters.get_scalar_view(("start_time",), ("World",), "s").get()
        #     == start_time
        # )
        # assert (
        #     in_parameters.get_scalar_view(("stop_time",), ("World",), "s").get()
        #     == stop_time
        # )
        # assert (
        #     in_parameters.get_scalar_view("ecs", ("World",), "K").get() == expected["ecs"]
        # )
        # assert (
        #     in_parameters.get_scalar_view(
        #         ("rf2xco2",), ("World",), "W / m^2"
        #     ).get()
        #     == expected["rf2xco2"]
        # )
        # np.testing.assert_allclose(
        #     tadapter._parameters.get_timeseries_view(("Emissions", "CO2"), ("World",), "GtCO2/yr", expected["emissions_time_points"], ParameterType.AVERAGE_TIMESERIES).get(),
        #     expected["emissions"],
        #     rtol=1e-10,
        #     atol=1e-15,
        # )

    def test_run(self, test_adapter, test_run_parameters):
        # TODO: add some tests about where input and output go to make sure adapter
        # isn't overwriting anything it's not meant to by accident
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters()
        test_adapter.reset()
        test_adapter.run()

    def test_step(self, test_adapter, test_run_parameters):
        # TODO: add some tests about where input and output go to make sure adapter
        # isn't overwriting anything it's not meant to by accident
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters()
        test_adapter.reset()
        assert test_adapter._current_time == test_run_parameters.start_time
        try:
            new_time = test_adapter.step()
            assert new_time > test_run_parameters.start_time
        except NotImplementedError:
            pytest.skip("Step unavailable for {}".format(type(tadapter)))

    # TODO: put this sort of test in issue about run function tests
    # def test_initialize_run_parameters_non_model_parameter(self, test_adapter):
    #     tname = ("junk",)
    #     test_adapter._parameters.get_writable_scalar_view(tname, ("World",), "K").set(4)
    #     error_msg = re.escape(
    #         "{} is not a {} parameter".format(tname[0], self.tadapter.__name__)
    #     )

    # with pytest.raises(NotAnScmParameterError, match=error_msg):
    #     test_adapter.initialize_run_parameters()

    @abstractmethod
    def test_openscm_standard_parameters_handling(self):
        """
        Test how the adapter handles OpenSCM's standard parameters.

        Implementers must implement this method to check what the user would get when
        OpenSCM's standard parameters are passed to the adapter. It might be that they
        get used, that they are re-mapped to a different name, that they are not
        supported and hence nothing is done. All these behaviours are valid, they just
        need to be tested and validated.

        We give an example of how such a test might look below.
        """
        pass  # TODO: implement once parameter usage can be checked

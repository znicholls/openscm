from abc import ABCMeta, abstractmethod

import pytest


class _AdapterTester(metaclass=ABCMeta):
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
    @abstractmethod
    def test_initialize(self, test_adapter):
        """
        Test the adapter is initiated as intended.

        Extra tests should be added for different adapters, to check any other
        expected behaviour as part of ``__init__`` calls.
        """
        test_adapter._initialize_model()

    @abstractmethod
    def test_shutdown(self, test_adapter):
        """
        Test the adapter can be shutdown.

        Extra tests should be added depending on what the adapter should actually
        do on shutdown.
        """
        del test_adapter

    @abstractmethod
    def test_initialize_model_input(self, test_adapter):
        """
        Test that initalizing model input does as intended.

        Extra tests should be added depending on what the adapter should actually
        do when model input is initialised.
        """
        assert not test_adapter._initialized
        test_adapter.initialize_model_input()
        assert test_adapter._initialized

    @abstractmethod
    def test_initialize_run_parameters(self, test_adapter, test_run_parameters):
        """
        Test that initalizing run parameters does as intended.

        Extra tests should be added depending on what the adapter should actually
        do when run parameters are initialised.
        """
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        assert not test_adapter._initialized
        test_adapter.initialize_run_parameters()
        assert test_adapter._initialized

    @abstractmethod
    def test_run(self, test_adapter, test_run_parameters):
        """
        Test that running the model does as intended.

        Extra tests should be added depending on what the adapter should actually
        do when run with the parameters provided by `test_run_parameters`.
        """
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters()
        test_adapter.reset()
        test_adapter.run()

    @abstractmethod
    def test_step(self, test_adapter, test_run_parameters):
        """
        Test that stepping the model does as intended.

        Extra tests should be added depending on what the adapter should do when
        stepped.
        """
        # if your model cannot step, override this method with the below
        """
        pytest.skip("Step unavailable for {}".format(type(test_adapter)))
        """
        # otherwise, use the below as a base
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters()
        test_adapter.reset()
        assert test_adapter._current_time == test_run_parameters.start_time

        new_time = test_adapter.step()
        assert new_time > test_run_parameters.start_time

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
        parameters = ParameterSet()
        parameters.generic("Start Time").value = np.datetime64("1850-01-01")
        parameters.generic("Stop Time").value = np.datetime64("2100-01-01")
        ecs_magnitude = 3.12
        parameters.scalar("Equilibrium Climate Sensitivity", "delta_degC").value = ecs_magnitude
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

        # From here onwards you can test whether e.g. the parameters have been used as
        # intended, an error was thrown or the parameters were not used.
        # If you're testing the parameters are used as intended, it might look
        # something like:
        assert (
            test_adapter._parameters.scalar(
                ("Model name", "model ecs parameter"), "delta_degC"
            ).value
            == ecs_magnitude
        )

        assert (
            output_parameters.scalar(
                "Equilibrium Climate Sensitivity", "delta_degC"
            ).value
            == ecs_magnitude
        )
        assert output_parameters.generic("Start Time").value == np.datetime64(
            "1850-01-01"
        )
        assert output_parameters.generic("Stop Time").value == np.datetime64(
            "2100-01-01"
        )

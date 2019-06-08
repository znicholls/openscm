"""
Module including all model adapters shipped with OpenSCM.
"""
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import numpy as np

from ..core.parameters import HierarchicalName
from ..core.parameterset import ParameterSet
from ..errors import AdapterNeedsModuleError

_loaded_adapters: Dict[str, type] = {}


class Adapter(metaclass=ABCMeta):
    """
    All model adapters in OpenSCM are implemented as subclasses of the
    :class:`openscm.adapter.Adapter` base class.

    :ref:`writing-adapters` provides a how-to on implementing an adapter.

    A model adapter is responsible for requesting the expected input parameters (in the
    expected time format and units) for the particular SCM from a
    :class:`openscm.core.ParameterSet`. It also runs its wrapped SCM and writes the
    output data back to a :class:`openscm.core.ParameterSet`.
    """

    _current_time: np.datetime64
    """Current time when using :func:`step`"""

    _initialized: bool
    """TODO: delete and replace with combination of reset and _update_model_parameter and _update_openscm_parameter"""

    _initialized_inputs: bool
    """TODO: delete and replace with combination of reset and _update_model_parameter and _update_openscm_parameter"""

    _output: ParameterSet
    """Output parameter set"""

    _parameters: ParameterSet
    """Input parameter set"""

    _parameters_versions: Dict[HierarchicalName, int]
    """Parameter versions the model is up to date with"""

    _openscm_standard_parameter_mappings: Dict[HierarchicalName, str]
    """
    Mapping from OpenSCM parameters to model parameters.

    If required, use property setters to add extra behaviour (like calculating a model 
    parameter based on the value of the OpenSCM parameter) when setting a model 
    parameter from an OpenSCM parameter.
    """

    def __init__(self, input_parameters: ParameterSet, output_parameters: ParameterSet):
        """
        Initialize the adapter as well as the model sitting underneath it.

		*Note:* as part of this process, all available model parameters are added to 
        ``input_parameters`` (if they're not already there).

        Parameters
        ----------
        input_parameters
            Input parameter set to use

        output_parameters
            Output parameter set to use
        """
        self._parameters = input_parameters
        self._output = output_parameters
        self._initialized = False  # TODO: delete and replace with combination of reset and _update_model_parameter and _update_openscm_parameter
        self._initialized_inputs = False  # TODO: delete and replace with combination of reset and _update_model_parameter and _update_openscm_parameter
        self._current_time = 0

    def __del__(self) -> None:
        """
        Destructor.
        """
        self._shutdown()

    def initialize_model_input(self) -> None:
        """
        TODO: delete and replace with combination of reset and _update_model_parameter and _update_openscm_parameter

        Initialize the model input.

        Called before the adapter is used in any way and at most once before a call to
        :func:`run` or :func:`step`.
        """
        if not self._initialized:
            self._initialize_model()
            self._initialized = True

        self._initialize_model_input()

    def initialize_run_parameters(self) -> None:
        """
        TODO: delete and replace with combination of reset and _update_model_parameter and _update_openscm_parameter

        Initialize parameters for the run.

        Called before the adapter is used in any way and at most once before a call to
        :func:`run` or :func:`step`.
        """
        if not self._initialized:
            self._initialize_model()
            self._initialized = True

        self._initialize_run_parameters()

    def reset(self) -> None:
        """
        Reset the model to prepare for a new run.

        Called once after each call of :func:`run` and to reset the model after several calls
        to :func:`step`.

        *Note:* this method sets the model configuration to match the values in
        `self._parameters``, which is not necessarily the same as the state which was 
        used at the start of the last run.
        """
        self._current_time = self._parameters.generic("Start Time").value
        self._reset()

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        self._run()

    def step(self) -> np.datetime64:
        """
        Do a single time step.

        Returns
        -------
        np.datetime64
            Current time
        """
        self._step()
        return self._current_time

    @abstractmethod
    def _initialize_model(self) -> None:
        """
        To be implemented by specific adapters.

        Initialize the model. Called only during :func:`__init__`.
        """

    @abstractmethod
    def _initialize_model_input(self) -> None:
        """
        TODO: delete and replace with combination of reset and _update_model_parameter and _update_openscm_parameter

        To be implemented by specific adapters.

        Initialize the model input. Called before the adapter is used in any way and at
        most once before a call to :func:`_run` or :func:`_step`.
        """

    @abstractmethod
    def _initialize_run_parameters(self) -> None:
        """
        TODO: delete and replace with combination of reset and _update_model_parameter and _update_openscm_parameter

        To be implemented by specific adapters.

        Initialize parameters for the run. Called before the adapter is used in any way
        and at most once before a call to :func:`_run` or :func:`_step`.
        """

    @abstractmethod
    def _reset(self) -> None:
        """
        TODO: delete and replace with combination of reset and _update_model_parameter and _update_openscm_parameter

        To be implemented by specific adapters.

        Reset the model to prepare for a new run. Called once after each call of
        :func:`_run` and to reset the model after several calls to :func:`_step`.
        """

    @abstractmethod
    def _run(self) -> None:
        """
        To be implemented by specific adapters.

        Run the model over the full time range.
        """

    @abstractmethod
    def _shutdown(self) -> None:
        """
        To be implemented by specific adapters.

        Shut the model down.
        """

    @abstractmethod
    def _step(self) -> None:
        """
        To be implemented by specific adapters.

        Do a single time step.
        """


def load_adapter(name: str) -> type:
    """
    Load adapter with a given name.

    Parameters
    ----------
    name
        Name of the adapter/model

    Returns
    -------
    type
        Requested adapter class

    Raises
    ------
    AdapterNeedsModuleError
        Adapter needs a module that is not installed
    KeyError
        Adapter/model not found
    """
    if name in _loaded_adapters:
        return _loaded_adapters[name]

    adapter: Optional[type] = None

    try:
        if name == "DICE":
            from .dice import DICE  # pylint: disable=cyclic-import

            adapter = DICE

        elif name == "PH99":
            from .ph99 import PH99  # pylint: disable=cyclic-import

            adapter = PH99

        """
        When implementing an additional adapter, include your adapter NAME here as:
        ```
        elif name == "NAME":
            from .NAME import NAME

            adapter = NAME
        ```
        """
    except ImportError:
        raise AdapterNeedsModuleError(
            "To run '{name}' you need to install additional dependencies. Please "
            "install them using `pip install openscm[model-{name}]`.".format(name=name)
        )

    if adapter is None:
        raise KeyError("Unknown model '{}'".format(name))

    _loaded_adapters[name] = adapter
    return adapter

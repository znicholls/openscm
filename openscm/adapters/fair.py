import warnings


import numpy as np
from fair.forward import fair_scm

from ..internal import Adapter
from ..models import PH99Model
from ..core import Core, ParameterSet
from ..units import unit_registry
from ..errors import NotAnScmParameterError
from ..parameters import _Parameter


class FAIR(Adapter):
    """Adapter for FaIR, https://github.com/OMS-NetZero/FAIR"""
    def __init__(self):
        self.model = None
        self.name = "FaIR"
        self._drivers = None
        self._config = None
        super().__init__()

    def initialize(self):
        super().initialize()
        self.model = fair_scm

    def set_drivers(self, core: Core) -> None:
        self._drivers = 3

    def set_config(self, parameters: ParameterSet) -> None:
        self._config = 12

    def run(self) -> None:
        res = fair_scm(**self._drivers, **self._config)

    def step(self) -> None:
        raise NotImplementedError("Although should be doable...")

    def shutdown(self) -> None:
        pass

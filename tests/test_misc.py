import numpy as np
import pytest


from openscm.units import unit_registry
from .conftest import assert_pint_equal


def test_pint_array_comparison():
    a = np.array([0, 2]) * unit_registry("GtC")
    b = np.array([0, 2]) * unit_registry("MtC")

    np.testing.assert_allclose(a, b)  # no error
    with pytest.raises(AssertionError):
        assert_pint_equal(a, b)

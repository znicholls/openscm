import warnings


import numpy as np
import pytest


from openscm.units import unit_registry


# temporary workaround until this is in Pint itself and can be imported
def assert_pint_equal(a, b, **kwargs):
    c = b.to(a.units)
    try:
        np.testing.assert_array_equal(a.magnitude, c.magnitude, **kwargs)

    except AssertionError as e:
        original_msg = "{}".format(e)
        note_line = "Note: values above have been converted to {}".format(a.units)
        units_lines = "Input units:\n" "x: {}\n" "y: {}".format(a.units, b.units)

        numerical_lines = (
            "Numerical values with units:\n" "x: {}\n" "y: {}".format(a, b)
        )

        error_msg = (
            "{}\n"
            "\n"
            "{}\n"
            "\n"
            "{}\n"
            "\n"
            "{}".format(original_msg, note_line, units_lines, numerical_lines)
        )

        raise AssertionError(error_msg)


def test_pint_array_comparison():
    a = np.array([0, 2]) * unit_registry("GtC")
    b = np.array([0, 2]) * unit_registry("MtC")

    # no error but does raise warning about stripping units
    with warnings.catch_warnings(record=True):
        np.testing.assert_allclose(a, b)

    # actually gives an error as we want
    with pytest.raises(AssertionError):
        assert_pint_equal(a, b)

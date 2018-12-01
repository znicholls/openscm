import traceback


import numpy as np


def assert_pint_equal(a, b):
    c = b.to(a.units)

    try:
        np.testing.assert_array_equal(a, c)
    except AssertionError as e:
        original_msg = "{}".format(e)
        note_line = "Note: values above have been converted to common units of {}".format(a.units)
        units_line = "Input units:\nx: {}\ny: {}".format(a.units, b.units)
        numerical_line = "Numerical values with units:\nx: {}\ny: {}".format(a, b)

        error_msg = "{}\n\n{}\n\n{}\n\n{}".format(original_msg, note_line, units_line, numerical_line)

        raise AssertionError(error_msg)

import numpy as np
import pytest

from openscm import timeseries_converter
from openscm.errors import InsufficientDataError
from openscm.parameters import ParameterType


def test_conversion_to_same_timeseries(combo):
    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.source,
        combo.source,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter._convert(
        combo.source_values, combo.source, combo.source
    )
    np.testing.assert_allclose(values, combo.source_values)
    assert timeseriesconverter.source_length == len(combo.source) - (
        1 if combo.timeseries_type == ParameterType.AVERAGE_TIMESERIES else 0
    )
    assert timeseriesconverter.target_length == len(combo.source) - (
        1 if combo.timeseries_type == ParameterType.AVERAGE_TIMESERIES else 0
    )


def test_insufficient_overlap(combo):
    with pytest.raises(InsufficientDataError):
        timeseries_converter.TimeseriesConverter(
            combo.source,
            combo.target - 1e6,
            combo.timeseries_type,
            combo.interpolation_type,
            combo.extrapolation_type,
        )


def test_conversion(combo):
    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.source,
        combo.target,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter._convert(
        combo.source_values, combo.source, combo.target
    )
    np.testing.assert_allclose(values, combo.target_values, atol=1e-10 * values.max())


def test_timeseriesconverter(combo):
    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.source,
        combo.target,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter.convert_from(combo.source_values)
    np.testing.assert_allclose(values, combo.target_values, atol=1e-10 * values.max())

    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.target,
        combo.source,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter.convert_to(combo.source_values)
    np.testing.assert_allclose(values, combo.target_values, atol=1e-10 * values.max())

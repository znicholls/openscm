"""
Fixtures and data for tests.
"""

from collections import namedtuple
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import pytest

from openscm import timeseries_converter
from openscm.core import ParameterSet
from openscm.parameters import ParameterType
from openscm.scmdataframe import ScmDataFrame
from openscm.utils import convert_openscm_time_to_datetime

try:
    from pyam import IamDataFrame
except ImportError:
    IamDataFrame = None

TEST_DATA = join(dirname(abspath(__file__)), "test_data")

TEST_DF_LONG_TIMES = pd.DataFrame(
    [
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy",
            "EJ/yr",
            1,
            6.0,
            6.0,
        ],
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy|Coal",
            "EJ/yr",
            0.5,
            3,
            3.0,
        ],
        [
            "a_model",
            "a_iam",
            "a_scenario2",
            "World",
            "Primary Energy",
            "EJ/yr",
            2,
            7,
            7,
        ],
    ],
    columns=[
        "climate_model",
        "model",
        "scenario",
        "region",
        "variable",
        "unit",
        datetime(1005, 1, 1),
        datetime(2010, 1, 1),
        datetime(3010, 12, 31),
    ],
)

TEST_DF = pd.DataFrame(
    [
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy",
            "EJ/yr",
            1,
            6.0,
            6.0,
        ],
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy|Coal",
            "EJ/yr",
            0.5,
            3,
            3.0,
        ],
        [
            "a_model",
            "a_iam",
            "a_scenario2",
            "World",
            "Primary Energy",
            "EJ/yr",
            2,
            7,
            7.0,
        ],
    ],
    columns=[
        "climate_model",
        "model",
        "scenario",
        "region",
        "variable",
        "unit",
        2005,
        2010,
        2015,
    ],
)

TEST_TS = np.array([[1, 6.0, 6], [0.5, 3, 3], [2, 7, 7]]).T


@pytest.fixture
def test_data_path():
    return TEST_DATA


@pytest.fixture
def doesnt_warn():
    @contextmanager
    def check_context():
        with pytest.warns(None) as record:
            yield
        if len(record):
            pytest.fail(
                "The following warnings were raised: {}".format(
                    [w.message for w in record.list]
                )
            )

    return check_context


@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF.copy()


@pytest.fixture(scope="function")
def test_scm_datetime_df():
    tdf = TEST_DF.copy()
    tdf.rename(
        {
            2005: datetime(2005, 6, 17, 12),
            2010: datetime(2010, 1, 3, 0),
            2015: datetime(2015, 1, 4, 0),
        },
        axis="columns",
        inplace=True,
    )

    yield ScmDataFrame(tdf)


@pytest.fixture(scope="function")
def test_ts():
    yield TEST_TS.copy()


@pytest.fixture(scope="function")
def test_iam_df():
    if IamDataFrame is None:
        pytest.skip("pyam is not installed")
    yield IamDataFrame(TEST_DF.copy())


@pytest.fixture(
    scope="function",
    params=[
        {"data": TEST_DF.copy()},
        pytest.param(
            {"data": IamDataFrame(TEST_DF.copy()).data},
            marks=pytest.mark.skipif(
                IamDataFrame is None, reason="pyam is not available"
            ),
        ),
        pytest.param(
            {"data": IamDataFrame(TEST_DF.copy()).timeseries()},
            marks=pytest.mark.skipif(
                IamDataFrame is None, reason="pyam is not available"
            ),
        ),
        {
            "data": TEST_TS.copy(),
            "columns": {
                "model": ["a_iam"],
                "climate_model": ["a_model"],
                "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
                "region": ["World"],
                "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
                "unit": ["EJ/yr"],
            },
            "index": [2005, 2010, 2015],
        },
    ],
)
def test_scm_df(request):
    if IamDataFrame is None:
        pytest.skip("pyam is not installed")
    yield ScmDataFrame(**request.param)


@pytest.fixture(scope="function")
def test_processing_scm_df():
    yield ScmDataFrame(
        data=np.array([[1, 6.0, 7], [0.5, 3, 2], [2, 7, 0], [-1, -2, 3]]).T,
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario", "a_scenario2", "a_scenario3"],
            "region": ["World"],
            "variable": [
                "Primary Energy",
                "Primary Energy|Coal",
                "Primary Energy",
                "Primary Energy",
            ],
            "unit": ["EJ/yr"],
        },
        index=[datetime(2005, 1, 1), datetime(2010, 1, 1), datetime(2015, 6, 12)],
    )


@pytest.fixture
def iamdf_type():
    return IamDataFrame


@pytest.fixture(scope="module")
def rcp26():
    fname = join(TEST_DATA, "rcp26_emissions.csv")
    return ScmDataFrame(fname)


def test_adapter(request):
    """
    Get an initialized instance of an the requesting classes ``tadapter`` property.
    """
    parameters = ParameterSet()
    parameters.get_writable_scalar_view(("ecs",), ("World",), "K").set(3)
    parameters.get_writable_scalar_view(("rf2xco2",), ("World",), "W / m^2").set(4.0)
    output_parameters = ParameterSet()
    try:
        yield request.cls.tadapter(parameters, output_parameters)
    except TypeError:
        pytest.skip("{} cannot be instantiated".format(str(request.cls.tadapter)))


@pytest.fixture
def assert_core():
    def check_func(expected, time, test_core, name, region, unit, time_points):
        pview = test_core.parameters.get_timeseries_view(
            name, region, unit, time_points, ParameterType.POINT_TIMESERIES
        )
        relevant_idx = (np.abs(time_points - time)).argmin()
        np.testing.assert_allclose(pview.get()[relevant_idx], expected)

    return check_func


@pytest.fixture(scope="function")
def test_run_parameters():
    RunParameters = namedtuple("RunParameters", ["start_time", "stop_time"])
    yield RunParameters(start_time=0, stop_time=100 * 365 * 24 * 60 * 60)


possible_source_values = [[1, 5, 3, 5, 7, 3, 2, 9]]

possible_target_values = [
    dict(
        source_start_time=0,
        source_period_length=10,
        target_start_time=-5,
        target_period_length=5,
        source_values=possible_source_values[0],
        target_values=[-1, 1, 3, 5, 4, 3, 4, 5, 6, 7, 5, 3, 2.5, 2, 5.5, 9, 12.5],
        timeseries_type=ParameterType.POINT_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.LINEAR,
    ),
    dict(
        source_start_time=0,
        source_period_length=10,
        target_start_time=-50,
        target_period_length=50,
        source_values=possible_source_values[0],
        target_values=[1, 1, 3, 9],
        timeseries_type=ParameterType.POINT_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.CONSTANT,
    ),
    dict(
        source_start_time=0,
        source_period_length=10,
        target_start_time=4,
        target_period_length=7,
        source_values=possible_source_values[0],
        target_values=[
            2.02142857,
            5.37857143,
            3.42857143,
            3.34285714,
            5.1,
            7.18571429,
            5.44642857,
            2.49642857,
            1.20714286,
            5.59285714,
            10.75,
        ],
        timeseries_type=ParameterType.AVERAGE_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.LINEAR,
    ),
    dict(
        source_start_time=0,
        source_period_length=10,
        target_start_time=0,
        target_period_length=5,
        source_values=possible_source_values[0],
        target_values=[
            0.0,
            2.0,
            4.75,
            5.25,
            3.0,
            3.0,
            4.5,
            5.5,
            7.25,
            6.75,
            3.625,
            2.375,
            1.25,
            2.75,
            7.25,
            10.75,
        ],
        timeseries_type=ParameterType.AVERAGE_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.LINEAR,
    ),
    dict(
        source_start_time=3,
        source_period_length=3,
        target_start_time=0,
        target_period_length=5,
        source_values=possible_source_values[0],
        target_values=[-1.66666667, 4.13333333, 4.13333333, 5.51666667, 3.01666667],
        timeseries_type=ParameterType.AVERAGE_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.LINEAR,
    ),
]

test_combinations = []

Combination = namedtuple(
    "TestCombination",
    [
        "source",
        "source_values",
        "target",
        "target_values",
        "timeseries_type",
        "interpolation_type",
        "extrapolation_type",
    ],
)
for index in possible_target_values:
    combination = Combination(
        source_values=np.array(index["source_values"]),
        source=timeseries_converter.create_time_points(
            index["source_start_time"],
            index["source_period_length"],
            len(index["source_values"]),
            index["timeseries_type"],
        ),
        target_values=np.array(index["target_values"]),
        target=timeseries_converter.create_time_points(
            index["target_start_time"],
            index["target_period_length"],
            len(index["target_values"]),
            index["timeseries_type"],
        ),
        timeseries_type=index["timeseries_type"],
        interpolation_type=index["interpolation_type"],
        extrapolation_type=index["extrapolation_type"],
    )
    test_combinations.append(combination)


@pytest.fixture(params=test_combinations)
def combo(request):
    return request.param


@pytest.fixture(params=test_combinations, scope="function")
def combo_df(request):
    combo = deepcopy(request.param)
    vals = combo._asdict()
    df_dts = [convert_openscm_time_to_datetime(d) for d in combo.source]

    # For average timeseries we drop the last time value so that the data and times are same length
    if combo.timeseries_type == ParameterType.AVERAGE_TIMESERIES:
        assert df_dts[-1] - df_dts[-2] == df_dts[-2] - df_dts[-3]
        df_dts = df_dts[:-1]
        vals["target"] = combo.target.copy()[:-1]
        assert len(vals["target"]) == len(combo.target_values)
        assert len(df_dts) == len(combo.source_values)

    df = ScmDataFrame(
        combo.source_values,
        columns={
            "scenario": ["a_scenario"],
            "model": ["a_model"],
            "region": ["World"],
            "variable": ["Emissions|BC"],
            "unit": ["Mg /yr"],
            "parameter_type": [
                (
                    "point"
                    if combo.timeseries_type == ParameterType.POINT_TIMESERIES
                    else "average"
                )
            ],
        },
        index=df_dts,
    )

    return Combination(**vals), df


def read_magicc7_variables():
    df = pd.read_csv(join(TEST_DATA, "magicc7_parameter_types.txt"), delimiter=",")

    return df.to_dict("records")


@pytest.fixture(scope="module", params=(read_magicc7_variables()))
def magicc7_variable(request):
    yield request.param

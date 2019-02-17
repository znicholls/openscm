import copy
import datetime
from dateutil import relativedelta
import re


import pytest
import numpy as np
import pandas as pd
from numpy import testing as npt
from pyam.core import (
    require_variable,
    categorize,
    filter_by_meta,
    validate,
    META_IDX,
    IamDataFrame,
)


from openscm.highlevel import (
    ScmDataFrame,
    convert_scmdataframe_to_core,
    convert_core_to_scmdataframe,
    df_append,
)
from openscm.scenarios import rcps
from openscm.constants import ONE_YEAR_IN_S_INTEGER
from openscm.utils import convert_datetime_to_openscm_time, round_to_nearest_year


from conftest import assert_core


def test_init_df_long_timespan(test_pd_longtime_df):
    df = ScmDataFrame(test_pd_longtime_df)

    pd.testing.assert_frame_equal(
        df.timeseries().reset_index(), test_pd_longtime_df, check_like=True
    )
    assert (df["year"].unique() == [1005, 3010]).all()
    assert df._data.index.name == "time"


def test_init_df_year_converted_to_datetime(test_pd_df):
    res = ScmDataFrame(test_pd_df)
    assert (res["year"].unique() == [2005, 2010]).all()
    assert (
        res["time"].unique()
        == [datetime.datetime(2005, 1, 1), datetime.datetime(2010, 1, 1)]
    ).all()


def get_test_pd_df_with_datetime_columns(tpdf):
    return tpdf.rename(
        {2005.0: datetime.datetime(2005, 1, 1), 2010.0: datetime.datetime(2010, 1, 1)},
        axis="columns",
    )


def test_init_ts(test_ts, test_pd_df):
    df = ScmDataFrame(
        test_ts,
        columns={
            "index": [2005, 2010],
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
            "region": ["World"],
            "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
            "unit": ["EJ/y"],
        },
    )

    tdf = get_test_pd_df_with_datetime_columns(test_pd_df)
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)

    b = ScmDataFrame(test_pd_df)

    pd.testing.assert_frame_equal(df.meta, b.meta, check_like=True)
    pd.testing.assert_frame_equal(df._data, b._data)


@pytest.mark.parametrize("years", [["2005.0", "2010.0"], ["2005", "2010"]])
def test_init_with_years_as_str(test_pd_df, years):
    df = copy.deepcopy(
        test_pd_df
    )  # This needs to be a deep copy so it doesn't break the other tests
    cols = copy.deepcopy(test_pd_df.columns.values)
    cols[-2:] = years
    df.columns = cols

    df = ScmDataFrame(df)

    obs = df._data.index
    exp = pd.Index(
        [datetime.datetime(2005, 1, 1), datetime.datetime(2010, 1, 1)],
        name="time",
        dtype="object",
    )
    pd.testing.assert_index_equal(obs, exp)


def test_col_order(test_scm_df):
    pd.testing.assert_index_equal(
        test_scm_df.meta.columns,
        pd.Index(["model", "scenario", "region", "variable", "unit", "climate_model"]),
    )


def test_init_ts_with_index(test_pd_df):
    df = ScmDataFrame(test_pd_df)
    tdf = get_test_pd_df_with_datetime_columns(test_pd_df)
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)


def test_init_with_decimal_years():
    inp_array = [2.0, 1.2, 7.9]
    d = pd.Series(inp_array, index=[1765.0, 1765.083, 1765.167])
    cols = {
        "model": ["a_model"],
        "scenario": ["a_scenario"],
        "region": ["World"],
        "variable": ["Primary Energy"],
        "unit": ["EJ/y"],
    }

    res = ScmDataFrame(d, columns=cols)
    assert (
        res["time"].unique()
        == [
            datetime.datetime(1765, 1, 1, 0, 0),
            datetime.datetime(1765, 1, 31, 7, 4, 48, 3),
            datetime.datetime(1765, 3, 2, 22, 55, 11, 999997),
        ]
    ).all()
    npt.assert_array_equal(res._data.loc[:, 0].values, inp_array)


def test_init_df_from_timeseries(test_scm_df):
    df = ScmDataFrame(test_scm_df.timeseries())
    pd.testing.assert_frame_equal(
        df.timeseries().reset_index(),
        test_scm_df.timeseries().reset_index(),
        check_like=True,
    )


def test_init_df_with_extra_col(test_pd_df):
    tdf = test_pd_df.copy()

    extra_col = "test value"
    extra_value = "scm_model"
    tdf[extra_col] = extra_value

    df = ScmDataFrame(tdf)

    tdf = get_test_pd_df_with_datetime_columns(tdf)
    assert extra_col in df.meta
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)


def test_init_datetime_subclass_long_timespan(test_pd_df):
    class TempSubClass(ScmDataFrame):
        def _format_datetime_col(self):
            # the subclass does not try to coerce the datetimes to pandas
            # datetimes, instead simply leaving the time column as object type,
            # so we don't run into the problem of pandas limited time period as
            # discussed in https://stackoverflow.com/a/37226672
            pass

    tdf = test_pd_df.copy()
    tmin = datetime.datetime(2005, 6, 17)
    tmax = datetime.datetime(3005, 6, 17)
    tdf = tdf.rename({2005: tmin, 2010: tmax}, axis="columns")

    df = TempSubClass(tdf)

    assert df["time"].max() == tmax
    assert df["time"].min() == tmin


def test_init_iam(test_iam_df, test_pd_df):
    a = ScmDataFrame(test_iam_df)
    b = ScmDataFrame(test_pd_df)

    pd.testing.assert_frame_equal(a.meta, b.meta)
    pd.testing.assert_frame_equal(a._data, b._data)


def test_init_self(test_iam_df):
    a = ScmDataFrame(test_iam_df)
    b = ScmDataFrame(a)

    pd.testing.assert_frame_equal(a.meta, b.meta)
    pd.testing.assert_frame_equal(a._data, b._data)


def test_as_iam(test_iam_df, test_pd_df):
    df = ScmDataFrame(test_pd_df).to_iamdataframe()

    assert isinstance(df, IamDataFrame)

    pd.testing.assert_frame_equal(test_iam_df.meta, df.meta)
    # we don't provide year column, fine as pyam are considering dropping year too
    tdf = df.data.copy()
    tdf["year"] = tdf["time"].apply(lambda x: x.year)
    tdf.drop("time", axis="columns", inplace=True)
    pd.testing.assert_frame_equal(test_iam_df.data, tdf, check_like=True)


def test_get_item(test_scm_df):
    assert test_scm_df["model"].unique() == ["a_iam"]


def test_variable_depth_0(test_scm_df):
    obs = list(test_scm_df.filter(level=0)["variable"].unique())
    exp = ["Primary Energy"]
    assert obs == exp


def test_variable_depth_0_keep_false(test_scm_df):
    obs = list(test_scm_df.filter(level=0, keep=False)["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_0_minus(test_scm_df):
    obs = list(test_scm_df.filter(level="0-")["variable"].unique())
    exp = ["Primary Energy"]
    assert obs == exp


def test_variable_depth_0_plus(test_scm_df):
    obs = list(test_scm_df.filter(level="0+")["variable"].unique())
    exp = ["Primary Energy", "Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1(test_scm_df):
    obs = list(test_scm_df.filter(level=1)["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1_minus(test_scm_df):
    obs = list(test_scm_df.filter(level="1-")["variable"].unique())
    exp = ["Primary Energy", "Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1_plus(test_scm_df):
    obs = list(test_scm_df.filter(level="1+")["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_raises(test_scm_df):
    pytest.raises(ValueError, test_scm_df.filter, level="1/")


def test_filter_error(test_scm_df):
    pytest.raises(ValueError, test_scm_df.filter, foo="foo")


def test_filter_year(test_scm_datetime_df):
    obs = test_scm_datetime_df.filter(year=2005)
    expected = datetime.datetime(2005, 6, 17, 12)

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_month", [6, "June", "Jun", "jun", ["Jun", "jun"]])
def test_filter_month(test_scm_datetime_df, test_month):
    obs = test_scm_datetime_df.filter(month=test_month)
    expected = datetime.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_month", [6, "Jun", "jun", ["Jun", "jun"]])
def test_filter_year_month(test_scm_datetime_df, test_month):
    obs = test_scm_datetime_df.filter(year=2005, month=test_month)
    expected = datetime.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_day", [17, "Fri", "Friday", "friday", ["Fri", "fri"]])
def test_filter_day(test_scm_datetime_df, test_day):
    obs = test_scm_datetime_df.filter(day=test_day)
    expected = datetime.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_hour", [0, 12, [12, 13]])
def test_filter_hour(test_scm_datetime_df, test_hour):
    obs = test_scm_datetime_df.filter(hour=test_hour)
    test_hour = [test_hour] if isinstance(test_hour, int) else test_hour
    expected_rows = test_scm_datetime_df["time"].apply(lambda x: x.hour).isin(test_hour)
    expected = test_scm_datetime_df["time"].loc[expected_rows].unique()

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected[0]


def test_filter_time_exact_match(test_scm_datetime_df):
    obs = test_scm_datetime_df.filter(time=datetime.datetime(2005, 6, 17, 12))
    expected = datetime.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


def test_filter_time_range(test_scm_datetime_df):
    error_msg = r".*datetime.datetime.*"
    with pytest.raises(TypeError, match=error_msg):
        test_scm_datetime_df.filter(
            year=range(datetime.datetime(2000, 6, 17), datetime.datetime(2009, 6, 17))
        )


def test_filter_time_range_year(test_scm_datetime_df):
    obs = test_scm_datetime_df.filter(year=range(2000, 2008))

    unique_time = obs["time"].unique()
    expected = datetime.datetime(2005, 6, 17, 12)

    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("month_range", [range(3, 7), "Mar-Jun"])
def test_filter_time_range_month(test_scm_datetime_df, month_range):
    obs = test_scm_datetime_df.filter(month=month_range)
    expected = datetime.datetime(2005, 6, 17, 12)

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("month_range", [["Mar-Jun", "Nov-Feb"]])
def test_filter_time_range_round_the_clock_error(test_scm_datetime_df, month_range):
    error_msg = re.escape(
        "string ranges must lead to increasing integer ranges, "
        "Nov-Feb becomes [11, 2]"
    )
    with pytest.raises(ValueError, match=error_msg):
        test_scm_datetime_df.filter(month=month_range)


@pytest.mark.parametrize("day_range", [range(14, 20), "Thu-Sat"])
def test_filter_time_range_day(test_scm_datetime_df, day_range):
    obs = test_scm_datetime_df.filter(day=day_range)
    expected = datetime.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("hour_range", [range(10, 14)])
def test_filter_time_range_hour(test_scm_datetime_df, hour_range):
    obs = test_scm_datetime_df.filter(hour=hour_range)

    expected_rows = (
        test_scm_datetime_df["time"].apply(lambda x: x.hour).isin(hour_range)
    )
    expected = test_scm_datetime_df["time"].loc[expected_rows].unique()

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected[0]


def test_filter_time_no_match(test_scm_datetime_df):
    obs = test_scm_datetime_df.filter(time=datetime.datetime(2004, 6, 18))
    assert obs._data.empty


def test_filter_time_not_datetime_error(test_scm_datetime_df):
    error_msg = re.escape("`time` can only be filtered by datetimes")
    with pytest.raises(TypeError, match=error_msg):
        test_scm_datetime_df.filter(time=2005)


def test_filter_time_not_datetime_range_error(test_scm_datetime_df):
    error_msg = re.escape("`time` can only be filtered by datetimes")
    with pytest.raises(TypeError, match=error_msg):
        test_scm_datetime_df.filter(time=range(2000, 2008))


def test_filter_as_kwarg(test_scm_df):
    obs = list(test_scm_df.filter(variable="Primary Energy|Coal")["scenario"].unique())
    assert obs == ["a_scenario"]


def test_filter_keep_false(test_scm_df):
    df = test_scm_df.filter(variable="Primary Energy|Coal", year=2005, keep=False)
    obs = df.filter(scenario="a_scenario").timeseries().values.ravel()
    npt.assert_array_equal(obs, [1, 6, np.nan, 3])


def test_filter_by_regexp(test_scm_df):
    obs = test_scm_df.filter(scenario="a_scenari.$", regexp=True)
    assert obs["scenario"].unique() == "a_scenario"


def test_filter_timeseries_different_length():
    df = ScmDataFrame(
        pd.DataFrame(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]]).T, index=[2000, 2001, 2002]
        ),
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario2"],
            "region": ["World"],
            "variable": ["Primary Energy"],
            "unit": ["EJ/y"],
        },
    )

    npt.assert_array_equal(
        df.filter(scenario="a_scenario2").timeseries().squeeze(), [4.0, 5.0]
    )
    npt.assert_array_equal(df.filter(year=2002).timeseries().squeeze(), 3.0)

    exp = pd.Series(["a_scenario"], name="scenario")
    obs = df.filter(year=2002)["scenario"]
    pd.testing.assert_series_equal(exp, obs)
    assert df.filter(scenario="a_scenario2", year=2002).timeseries().empty


def test_timeseries(test_scm_df):
    dct = {
        "model": ["a_model"] * 2,
        "scenario": ["a_scenario"] * 2,
        "years": [2005, 2010],
        "value": [1, 6],
    }
    exp = pd.DataFrame(dct).pivot_table(
        index=["model", "scenario"], columns=["years"], values="value"
    )
    obs = test_scm_df.filter(
        variable="Primary Energy", scenario="a_scenario"
    ).timeseries()
    npt.assert_array_equal(obs, exp)


def test_timeseries_meta(test_scm_df):
    obs = test_scm_df.filter(variable="Primary Energy").timeseries(
        meta=["scenario", "model"]
    )
    npt.assert_array_equal(obs.index.names, ["scenario", "model"])


def test_timeseries_duplicated(test_scm_df):
    pytest.raises(ValueError, test_scm_df.timeseries, meta=["scenario"])


def test_quantile_over_lower(test_processing_scm_df):
    # not sure how this should work in place, in particular what do you fill
    # the column which has been 'quantiled over' with? The value of the quantile
    # e.g. 55th percentile?
    exp = pd.DataFrame(
        [
            ["a_model", "a_iam", "World", "Primary Energy", "EJ/y", -1, -2, 3],
            ["a_model", "a_iam", "World", "Primary Energy|Coal", "EJ/y", 0.5, 3, 2],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            datetime.datetime(2005, 1, 1, 0, 0, 0),
            datetime.datetime(2010, 1, 1, 0, 0, 0),
            datetime.datetime(2015, 1, 1, 0, 0, 0),
        ],
    )
    obs = test_processing_scm_df.quantile_over("scenario", 0)
    pd.testing.assert_frame_equal(
        exp.reorder_levels(obs.index.names), obs, check_like=True
    )


def test_quantile_over_upper(test_processing_scm_df):
    # not sure how this should work in place, in particular what do you fill
    # the column which has been 'quantiled over' with? The value of the quantile
    # e.g. 55th percentile?
    exp = pd.DataFrame(
        [
            ["a_model", "a_iam", "World", "Primary Energy", "EJ/y", 2, 7, 7],
            ["a_model", "a_iam", "World", "Primary Energy|Coal", "EJ/y", 0.5, 3, 2],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            datetime.datetime(2005, 1, 1, 0, 0, 0),
            datetime.datetime(2010, 1, 1, 0, 0, 0),
            datetime.datetime(2015, 1, 1, 0, 0, 0),
        ],
    )
    obs = test_processing_scm_df.quantile_over("scenario", 1)
    pd.testing.assert_frame_equal(
        exp.reorder_levels(obs.index.names), obs, check_like=True
    )


def test_mean_over(test_processing_scm_df):
    # not sure how this should work in place, in particular what do you fill
    # the column which has been 'quantiled over' with? The value of the quantile
    # e.g. 55th percentile?
    exp = pd.DataFrame(
        [
            [
                "a_model",
                "a_iam",
                "World",
                "Primary Energy",
                "EJ/y",
                2 / 3,
                11 / 3,
                10 / 3,
            ],
            ["a_model", "a_iam", "World", "Primary Energy|Coal", "EJ/y", 0.5, 3, 2],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            datetime.datetime(2005, 1, 1, 0, 0, 0),
            datetime.datetime(2010, 1, 1, 0, 0, 0),
            datetime.datetime(2015, 1, 1, 0, 0, 0),
        ],
    )
    obs = test_processing_scm_df.mean_over("scenario")
    pd.testing.assert_frame_equal(
        exp.reorder_levels(obs.index.names), obs, check_like=True
    )


def test_median_over(test_processing_scm_df):
    # not sure how this should work in place, in particular what do you fill
    # the column which has been 'quantiled over' with? The value of the quantile
    # e.g. 55th percentile?
    exp = pd.DataFrame(
        [
            ["a_model", "a_iam", "World", "Primary Energy", "EJ/y", 1, 6, 3],
            ["a_model", "a_iam", "World", "Primary Energy|Coal", "EJ/y", 0.5, 3, 2],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            datetime.datetime(2005, 1, 1, 0, 0, 0),
            datetime.datetime(2010, 1, 1, 0, 0, 0),
            datetime.datetime(2015, 1, 1, 0, 0, 0),
        ],
    )
    obs = test_processing_scm_df.median_over("scenario")
    pd.testing.assert_frame_equal(
        exp.reorder_levels(obs.index.names), obs, check_like=True
    )


def test_relative_to_ref_period_mean(test_processing_scm_df):
    # not sure how this should work in place, in particular what do you fill
    # the column that is now relative to with? A first implementation is here
    # but it might not be the most sensible.
    exp = pd.DataFrame(
        [
            [
                "a_model",
                "a_iam",
                "a_scenario",
                "World",
                "Primary Energy (2005-2010 ref. period)",
                "EJ/y",
                -2.5,
                2.5,
                3.5,
            ],
            [
                "a_model",
                "a_iam",
                "a_scenario",
                "World",
                "Primary Energy|Coal (2005-2010 ref. period)",
                "EJ/y",
                -1.25,
                1.25,
                0.25,
            ],
            [
                "a_model",
                "a_iam",
                "a_scenario2",
                "World",
                "Primary Energy|Coal (2005-2010 ref. period)",
                "EJ/y",
                -2.5,
                2.5,
                -4.5,
            ],
            [
                "a_model",
                "a_iam",
                "a_scenario3",
                "World",
                "Primary Energy|Coal (2005-2010 ref. period)",
                "EJ/y",
                0.5,
                -0.5,
                4.5,
            ],
        ],
        columns=[
            "climate_model",
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
            datetime.datetime(2005, 1, 1, 0, 0, 0),
            datetime.datetime(2010, 1, 1, 0, 0, 0),
            datetime.datetime(2015, 1, 1, 0, 0, 0),
        ],
    )
    # what to do if ref period does not line up with provided data?
    obs = test_processing_scm_df.relative_to_ref_period_mean(
        (datetime.datetime(2005, 1, 1, 0, 0, 0), datetime.datetime(2010, 1, 1, 0, 0, 0))
    )
    pd.testing.assert_frame_equal(
        exp.reorder_levels(obs.index.names), obs, check_like=True
    )


@pytest.mark.skip
def test_require_variable(test_scm_df):
    obs = test_scm_df.require_variable(
        variable="Primary Energy|Coal", exclude_on_fail=True
    )
    assert len(obs) == 1
    assert obs.loc[0, "scenario"] == "a_scenario2"

    assert list(test_scm_df["exclude"]) == [False, True]


@pytest.mark.skip
def test_require_variable_top_level(test_scm_df):
    obs = require_variable(
        test_scm_df, variable="Primary Energy|Coal", exclude_on_fail=True
    )
    assert len(obs) == 1
    assert obs.loc[0, "scenario"] == "a_scenario2"

    assert list(test_scm_df["exclude"]) == [False, True]


@pytest.mark.skip
def test_validate_all_pass(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy": {"up": 10}}, exclude_on_fail=True)
    assert obs is None
    assert len(test_scm_df.data) == 6  # data unchanged

    assert list(test_scm_df["exclude"]) == [False, False]  # none excluded


@pytest.mark.skip
def test_validate_nonexisting(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy|Coal": {"up": 2}}, exclude_on_fail=True)
    assert len(obs) == 1
    assert obs["scenario"].values[0] == "a_scenario"

    assert list(test_scm_df["exclude"]) == [True, False]  # scenario with failed
    # validation excluded, scenario with non-defined value passes validation


@pytest.mark.skip
def test_validate_up(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy": {"up": 6.5}}, exclude_on_fail=False)
    assert len(obs) == 1
    assert obs["year"].values[0] == 2010

    assert list(test_scm_df["exclude"]) == [False, False]  # assert none excluded


@pytest.mark.skip
def test_validate_lo(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy": {"up": 8, "lo": 2.0}})
    assert len(obs) == 1
    assert obs["year"].values[0] == 2005
    assert list(obs["scenario"].values) == ["a_scenario"]


@pytest.mark.skip
def test_validate_both(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy": {"up": 6.5, "lo": 2.0}})
    assert len(obs) == 2
    assert list(obs["year"].values) == [2005, 2010]
    assert list(obs["scenario"].values) == ["a_scenario", "a_scenario2"]


@pytest.mark.skip
def test_validate_year(test_scm_df):
    obs = test_scm_df.validate(
        {"Primary Energy": {"up": 5.0, "year": 2005}}, exclude_on_fail=False
    )
    assert obs is None

    obs = test_scm_df.validate(
        {"Primary Energy": {"up": 5.0, "year": 2010}}, exclude_on_fail=False
    )
    assert len(obs) == 2


@pytest.mark.skip
def test_validate_exclude(test_scm_df):
    test_scm_df.validate({"Primary Energy": {"up": 6.0}}, exclude_on_fail=True)
    assert list(test_scm_df["exclude"]) == [False, True]


@pytest.mark.skip
def test_validate_top_level(test_scm_df):
    obs = validate(
        test_scm_df,
        criteria={"Primary Energy": {"up": 6.0}},
        exclude_on_fail=True,
        variable="Primary Energy",
    )
    assert len(obs) == 1
    assert obs["year"].values[0] == 2010
    assert list(test_scm_df["exclude"]) == [False, True]


@pytest.mark.skip
def test_category_none(test_scm_df):
    test_scm_df.categorize("category", "Testing", {"Primary Energy": {"up": 0.8}})
    assert "category" not in test_scm_df.meta.columns


@pytest.mark.skip
def test_category_pass(test_scm_df):
    dct = {
        "model": ["a_model", "a_model"],
        "scenario": ["a_scenario", "a_scenario2"],
        "category": ["foo", None],
    }
    exp = pd.DataFrame(dct).set_index(["model", "scenario"])["category"]

    test_scm_df.categorize(
        "category", "foo", {"Primary Energy": {"up": 6, "year": 2010}}
    )
    obs = test_scm_df["category"]
    pd.testing.assert_series_equal(obs, exp)


@pytest.mark.skip
def test_category_top_level(test_scm_df):
    dct = {
        "model": ["a_model", "a_model"],
        "scenario": ["a_scenario", "a_scenario2"],
        "category": ["Testing", None],
    }
    exp = pd.DataFrame(dct).set_index(["model", "scenario"])["category"]

    categorize(
        test_scm_df,
        "category",
        "Testing",
        criteria={"Primary Energy": {"up": 6, "year": 2010}},
        variable="Primary Energy",
    )
    obs = test_scm_df["category"]
    pd.testing.assert_series_equal(obs, exp)


def test_append(test_scm_df):
    test_scm_df.set_meta([5, 6, 7], name="col1")
    other = test_scm_df.filter(scenario="a_scenario2").rename(
        {"variable": {"Primary Energy": "Primary Energy clone"}}
    )

    other.set_meta(2, name="col1")
    other.set_meta("b", name="col2")

    df = test_scm_df.append(other)
    assert isinstance(df, ScmDataFrame)

    # check that the new meta.index is updated, but not the original one
    assert "col1" in test_scm_df.meta

    # assert that merging of meta works as expected
    npt.assert_array_equal(
        df.meta.sort_values(["scenario", "variable"])["col1"].values, [5, 6, 7, 2]
    )
    pd.testing.assert_series_equal(
        df.meta.sort_values(["scenario", "variable"])["col2"].reset_index(drop=True),
        pd.Series([np.nan, np.nan, np.nan, "b"]),
        check_names=False,
    )

    # assert that appending data works as expected
    ts = df.timeseries().sort_index()
    npt.assert_array_equal(ts.iloc[2], ts.iloc[3])
    pd.testing.assert_index_equal(
        df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "col1",
                "col2",
            ]
        ),
    )


def test_append_exact_duplicates(test_scm_df):
    other = copy.deepcopy(test_scm_df)
    test_scm_df.append(other).timeseries()

    pd.testing.assert_frame_equal(test_scm_df.timeseries(), other.timeseries())


def test_append_duplicates(test_scm_df):
    other = copy.deepcopy(test_scm_df)
    other["time"] = [2020, 2030]
    other._format_datetime_col()

    res = test_scm_df.append(other)

    obs = res.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2.0, 7.0, 2.0, 7.0]
    npt.assert_almost_equal(obs, exp)


def test_append_duplicate_times(test_scm_df):
    other = copy.deepcopy(test_scm_df)
    other._data *= 2

    res = test_scm_df.append(other)

    obs = res.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [(2.0 + 4.0) / 2, (7.0 + 14.0) / 2]
    npt.assert_almost_equal(obs, exp)


def test_append_inplace(test_scm_df):
    other = copy.deepcopy(test_scm_df)
    other._data *= 2

    obs = test_scm_df.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2, 7]
    npt.assert_almost_equal(obs, exp)

    test_scm_df.append(other, inplace=True)

    obs = test_scm_df.filter(scenario="a_scenario2").timeseries().squeeze()
    # is this averaging business really what we want
    exp = [(2.0 + 4.0) / 2, (7.0 + 14.0) / 2]
    npt.assert_almost_equal(obs, exp)


def get_append_col_order_time_dfs(base):
    other_2 = base.filter(variable="Primary Energy|Coal")
    base.set_meta("co2_only", name="runmodus")
    other = copy.deepcopy(base)

    tnew_var = "Primary Energy|Gas"
    other._meta = other._meta[sorted(other._meta.columns.values)]
    other._meta.loc[1, "variable"] = tnew_var

    tdata = other._data.copy().reset_index()
    tdata["time"] = [
        datetime.datetime(2002, 1, 1, 0, 0),
        datetime.datetime(2008, 1, 1, 0, 0),
    ]
    tdata = tdata.set_index("time")
    tdata.index = tdata.index.astype("object")

    other._data = tdata

    other_2._meta["ecs"] = 3.0
    other_2._meta["climate_model"] = "a_model2"

    exp = ScmDataFrame(
        pd.DataFrame(
            np.array(
                [
                    [1.0, 1.0, 6.0, 6.0],
                    [np.nan, 0.5, np.nan, 3.0],
                    [0.5, np.nan, 3.0, np.nan],
                    [2.0, 2.0, 7.0, 7.0],
                    [np.nan, 0.5, np.nan, 3.0],
                ]
            ).T,
            index=[2002, 2005, 2008, 2010],
        ),
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model", "a_model", "a_model", "a_model", "a_model2"],
            "scenario": [
                "a_scenario",
                "a_scenario",
                "a_scenario",
                "a_scenario2",
                "a_scenario",
            ],
            "region": ["World"],
            "variable": [
                "Primary Energy",
                "Primary Energy|Coal",
                "Primary Energy|Gas",
                "Primary Energy",
                "Primary Energy|Coal",
            ],
            "unit": ["EJ/y"],
            "runmodus": ["co2_only", "co2_only", "co2_only", "co2_only", np.nan],
            "ecs": [np.nan, np.nan, np.nan, np.nan, 3.0],
        },
    )

    return base, other, other_2, exp


def test_append_column_order_time_interpolation(test_scm_df):
    base, other, other_2, exp = get_append_col_order_time_dfs(test_scm_df)

    res = df_append([test_scm_df, other, other_2])

    pd.testing.assert_frame_equal(
        res.timeseries().sort_index(),
        exp.timeseries().reorder_levels(res.timeseries().index.names).sort_index(),
        check_like=True,
    )


def test_append_chain_column_order_time_interpolation(test_scm_df):
    base, other, other_2, exp = get_append_col_order_time_dfs(test_scm_df)

    res = test_scm_df.append(other).append(other_2)

    pd.testing.assert_frame_equal(
        res.timeseries().sort_index(),
        exp.timeseries().reorder_levels(res.timeseries().index.names).sort_index(),
        check_like=True,
    )


def test_append_inplace_column_order_time_interpolation(test_scm_df):
    base, other, other_2, exp = get_append_col_order_time_dfs(test_scm_df)

    test_scm_df.append(other, inplace=True)
    test_scm_df.append(other_2, inplace=True)

    pd.testing.assert_frame_equal(
        test_scm_df.timeseries().sort_index(),
        exp.timeseries()
        .reorder_levels(test_scm_df.timeseries().index.names)
        .sort_index(),
        check_like=True,
    )


def test_append_inplace_preexisinting_nan(test_scm_df):
    other = copy.deepcopy(test_scm_df)
    other._data *= 2
    other._meta["climate_model"] = "a_model2"
    other.set_meta(np.nan, name="junk")

    original_ts = test_scm_df.timeseries().copy()
    res = test_scm_df.append(other)

    # make sure underlying hasn't changed when not appending inplace
    pd.testing.assert_frame_equal(original_ts, test_scm_df.timeseries())

    exp = pd.concat([test_scm_df.timeseries(), other.timeseries()])
    exp["junk"] = np.nan
    exp.set_index("junk", append=True, inplace=True)

    pd.testing.assert_frame_equal(
        res.timeseries().reorder_levels(exp.index.names).sort_index().reset_index(),
        exp.sort_index().reset_index(),
        check_like=True,
    )


@pytest.mark.skip
def test_interpolate(test_scm_df):
    test_scm_df.interpolate(2007)
    dct = {
        "model": ["a_model"] * 3,
        "scenario": ["a_scenario"] * 3,
        "years": [2005, 2007, 2010],
        "value": [1, 3, 6],
    }
    exp = pd.DataFrame(dct).pivot_table(
        index=["model", "scenario"], columns=["years"], values="value"
    )
    variable = {"variable": "Primary Energy"}
    obs = test_scm_df.filter(**variable).timeseries()
    npt.assert_array_equal(obs, exp)

    # redo the inpolation and check that no duplicates are added
    test_scm_df.interpolate(2007)
    assert not test_scm_df.filter(**variable).data.duplicated().any()


def test_set_meta_no_name(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_iam"], ["World"]],
        codes=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )
    s = pd.Series(data=[0.3], index=idx)
    pytest.raises(ValueError, test_scm_df.set_meta, s)


def test_set_meta_as_named_series(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_iam"], ["World"]],
        codes=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )

    s = pd.Series(data=[0.3], index=idx)
    s.name = "meta_values"
    test_scm_df.set_meta(s)

    exp = pd.Series(data=[0.3, 0.3, np.nan], index=test_scm_df.meta.index)
    exp.name = "meta_values"

    obs = test_scm_df["meta_values"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_values",
            ]
        ),
    )


def test_set_meta_as_unnamed_series(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_iam"], ["World"]],
        codes=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )

    s = pd.Series(data=[0.3], index=idx)
    test_scm_df.set_meta(s, name="meta_values")

    exp = pd.Series(data=[0.3, 0.3, np.nan], index=test_scm_df.meta.index)
    exp.name = "meta_values"

    obs = test_scm_df["meta_values"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_values",
            ]
        ),
    )


def test_set_meta_non_unique_index_fail(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_iam"], ["a_scenario"], ["a", "b"]],
        codes=[[0, 0], [0, 0], [0, 1]],
        names=["model", "scenario", "region"],
    )
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, test_scm_df.set_meta, s)


def test_set_meta_non_existing_index_fail(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_iam", "fail_model"], ["a_scenario", "fail_scenario"]],
        codes=[[0, 1], [0, 1]],
        names=["model", "scenario"],
    )
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, test_scm_df.set_meta, s)


def test_set_meta_by_df(test_scm_df):
    df = pd.DataFrame(
        [["a_iam", "a_scenario", "World", 1]],
        columns=["model", "scenario", "region", "col"],
    )

    test_scm_df.set_meta(meta=0.3, name="meta_values", index=df)

    exp = pd.Series(data=[0.3, 0.3, np.nan], index=test_scm_df.meta.index)
    exp.name = "meta_values"

    obs = test_scm_df["meta_values"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_values",
            ]
        ),
    )


def test_set_meta_as_series(test_scm_df):
    # TODO: This is a bit yucky. You can pass in a series which isnt the same length as the meta table without warning
    s = pd.Series([0.3, 0.4])
    test_scm_df.set_meta(s, "meta_series")

    exp = pd.Series(data=[0.3, 0.4, np.nan], index=test_scm_df.meta.index)
    exp.name = "meta_series"

    obs = test_scm_df["meta_series"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_series",
            ]
        ),
    )


def test_set_meta_as_int(test_scm_df):
    test_scm_df.set_meta(3.2, "meta_int")

    exp = pd.Series(data=[3.2, 3.2, 3.2], index=test_scm_df.meta.index, name="meta_int")

    obs = test_scm_df["meta_int"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_int",
            ]
        ),
    )


def test_set_meta_as_str(test_scm_df):
    test_scm_df.set_meta("testing", name="meta_str")

    exp = pd.Series(
        data=["testing", "testing", "testing"],
        index=test_scm_df.meta.index,
        name="meta_str",
    )

    obs = test_scm_df["meta_str"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_str",
            ]
        ),
    )


def test_set_meta_as_str_list(test_scm_df):
    test_scm_df.set_meta(["testing", "testing2", "testing2"], name="category")
    obs = test_scm_df.filter(category="testing")
    assert obs["scenario"].unique() == "a_scenario"


def test_set_meta_as_str_by_index(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_iam"], ["a_scenario"]],
        codes=[[0], [0]],
        names=["model", "scenario"],
    )

    test_scm_df.set_meta("foo", "meta_str", idx)

    obs = pd.Series(test_scm_df["meta_str"].values)
    pd.testing.assert_series_equal(obs, pd.Series(["foo", "foo", None]))
    pd.testing.assert_index_equal(
        test_scm_df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_str",
            ]
        ),
    )


def test_filter_by_bool(test_scm_df):
    test_scm_df.set_meta([True, False, False], name="exclude")
    obs = test_scm_df.filter(exclude=True)
    assert obs["scenario"].unique() == "a_scenario"


def test_filter_by_int(test_scm_df):
    test_scm_df.set_meta([1, 2, 3], name="test")
    obs = test_scm_df.filter(test=[1])
    assert obs["scenario"].unique() == "a_scenario"


def test_rename_variable(test_scm_df):
    mapping = {
        "variable": {
            "Primary Energy": "Primary Energy|Total",
            "Primary Energy|Coal": "Primary Energy|Fossil",
        }
    }

    obs = test_scm_df.rename(mapping)

    exp = pd.Series(
        ["Primary Energy|Total", "Primary Energy|Fossil", "Primary Energy|Total"]
    )
    pd.testing.assert_series_equal(
        obs["variable"], exp, check_index_type=False, check_names=False
    )


def test_rename_index_fail(test_scm_df):
    mapping = {"scenario": {"a_scenario": "a_scenario2"}}
    pytest.raises(ValueError, test_scm_df.rename, mapping)


@pytest.mark.skip
def test_convert_unit():
    df = ScmDataFrame(
        pd.DataFrame(
            [
                ["model", "scen", "SST", "test_1", "A", 1, 5],
                ["model", "scen", "SDN", "test_2", "unit", 2, 6],
                ["model", "scen", "SST", "test_3", "C", 3, 7],
            ],
            columns=["model", "scenario", "region", "variable", "unit", 2005, 2010],
        )
    )

    unit_conv = {"A": ["B", 5], "C": ["D", 3]}

    obs = df.convert_unit(unit_conv).data.reset_index(drop=True)

    exp = ScmDataFrame(
        pd.DataFrame(
            [
                ["model", "scen", "SST", "test_1", "B", 5, 25],
                ["model", "scen", "SDN", "test_2", "unit", 2, 6],
                ["model", "scen", "SST", "test_3", "D", 9, 21],
            ],
            columns=["model", "scenario", "region", "variable", "unit", 2005, 2010],
        )
    ).data.reset_index(drop=True)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


@pytest.mark.skip
def test_pd_filter_by_meta(test_scm_df):
    data = df_filter_by_meta_matching_idx.set_index(["model", "region"])

    test_scm_df.set_meta([True, False], "boolean")
    test_scm_df.set_meta(0, "integer")

    obs = filter_by_meta(data, test_scm_df, join_meta=True, boolean=True, integer=None)
    obs = obs.reindex(columns=["scenario", "col", "boolean", "integer"])

    exp = data.iloc[0:2].copy()
    exp["boolean"] = True
    exp["integer"] = 0

    pd.testing.assert_frame_equal(obs, exp)


@pytest.mark.skip
def test_pd_filter_by_meta_no_index(test_scm_df):
    data = df_filter_by_meta_matching_idx

    test_scm_df.set_meta([True, False], "boolean")
    test_scm_df.set_meta(0, "int")

    obs = filter_by_meta(data, test_scm_df, join_meta=True, boolean=True, int=None)
    obs = obs.reindex(columns=META_IDX + ["region", "col", "boolean", "int"])

    exp = data.iloc[0:2].copy()
    exp["boolean"] = True
    exp["int"] = 0

    pd.testing.assert_frame_equal(obs, exp)


@pytest.mark.skip
def test_pd_filter_by_meta_nonmatching_index(test_scm_df):
    data = df_filter_by_meta_nonmatching_idx
    test_scm_df.set_meta(["a", "b"], "string")

    obs = filter_by_meta(data, test_scm_df, join_meta=True, string="b")
    obs = obs.reindex(columns=["scenario", 2010, 2020, "string"])

    exp = data.iloc[2:3].copy()
    exp["string"] = "b"

    pd.testing.assert_frame_equal(obs, exp)


@pytest.mark.skip
def test_pd_join_by_meta_nonmatching_index(test_scm_df):
    data = df_filter_by_meta_nonmatching_idx
    test_scm_df.set_meta(["a", "b"], "string")

    obs = filter_by_meta(data, test_scm_df, join_meta=True, string=None)
    obs = obs.reindex(columns=["scenario", 2010, 2020, "string"])

    exp = data.copy()
    exp["string"] = [np.nan, np.nan, "b"]

    pd.testing.assert_frame_equal(obs.sort_index(level=1), exp)


def test_convert_scmdataframe_to_core():
    tdata = rcps.filter(scenario="RCP26")

    res = convert_scmdataframe_to_core(tdata)

    tstart_dt = tdata["time"].min()
    tstart = convert_datetime_to_openscm_time(tstart_dt)
    tperiod_length = ONE_YEAR_IN_S_INTEGER

    def get_comparison_time_for_year(yr):
        return convert_datetime_to_openscm_time(
            tstart_dt + relativedelta.relativedelta(years=yr - tstart_dt.year)
        )

    assert_core(
        9.14781,
        get_comparison_time_for_year(2017),
        res,
        ("Emissions", "CO2", "MAGICC Fossil and Industrial"),
        "World",
        "GtC / yr",
        tstart,
        tperiod_length,
    )

    assert_core(
        6.124 + 1.2981006,
        get_comparison_time_for_year(1993),
        res,
        ("Emissions", "CO2"),
        "World",
        "GtC / yr",
        tstart,
        tperiod_length,
    )

    assert_core(
        7.2168971,
        get_comparison_time_for_year(1983),
        res,
        ("Emissions", "N2O"),
        "World",
        "MtN2ON / yr",
        tstart,
        tperiod_length,
    )

    assert_core(
        0.56591996,
        get_comparison_time_for_year(1766),
        res,
        ("Emissions", "OC"),
        "World",
        "MtOC / yr",
        tstart,
        tperiod_length,
    )

    assert_core(
        0.22445,
        get_comparison_time_for_year(2087),
        res,
        ("Emissions", "SF6"),
        "World",
        "ktSF6 / yr",
        tstart,
        tperiod_length,
    )


def test_convert_core_to_scmdataframe():
    tdata = rcps.filter(scenario="RCP26")

    intermediate = convert_scmdataframe_to_core(tdata)

    res = convert_core_to_scmdataframe(
        intermediate,
        period_length=ONE_YEAR_IN_S_INTEGER,
        model="IMAGE",
        scenario="RCP26",
        climate_model="unspecified",
    )

    # necessary as moving from even timesteps in seconds does not match perfectly with
    # yearly timesteps (which are not always the same number of seconds apart due to
    # leap years)
    tdata["time"] = tdata["time"].apply(round_to_nearest_year)
    res["time"] = res["time"].apply(round_to_nearest_year)

    pd.testing.assert_frame_equal(
        tdata.timeseries().reset_index(),
        res.timeseries().reset_index(),
        check_like=True,
    )


def test_resample(test_scm_df):
    res = test_scm_df.resample("AS").interpolate()

    obs = (
        res.filter(scenario="a_scenario", variable="Primary Energy")
        .timeseries()
        .T.squeeze()
    )
    exp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    npt.assert_almost_equal(obs, exp, decimal=1)


def test_resample_long_datetimes(test_pd_longtime_df):
    from cftime import DatetimeGregorian

    df = ScmDataFrame(test_pd_longtime_df)
    res = df.resample("AS").interpolate()

    assert res.timeseries().T.index[0] == DatetimeGregorian(1005, 1, 1)
    assert res.timeseries().T.index[-1] == DatetimeGregorian(3010, 1, 1)

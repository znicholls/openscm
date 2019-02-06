import datetime
from dateutil import relativedelta


import pytest
import pandas as pd


from openscm.highlevel import (
    ScmDataFrame,
    convert_scmdataframe_to_core,
    convert_core_to_scmdataframe,
)
from openscm.scenarios import rcps
from openscm.constants import ONE_YEAR_IN_S_INTEGER
from openscm.utils import convert_datetime_to_openscm_time


from conftest import assert_core


def test_init_df_long_timespan(test_pd_df):
    df = ScmDataFrame(test_pd_df)

    pd.testing.assert_frame_equal(
        df.timeseries().reset_index(), test_pd_df, check_like=True
    )


def test_init_df_datetime_error(test_pd_df):
    tdf = ScmDataFrame(test_pd_df).data
    tdf["time"] = 2010

    error_msg = r"^All time values must be convertible to datetime\. The following values are not:(.|\s)*$"
    with pytest.raises(ValueError, match=error_msg):
        ScmDataFrame(tdf)


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


def test_convert_scmdataframe_to_core():
    tdata = rcps.filter(scenario="RCP26")

    intermediate = convert_scmdataframe_to_core(tdata)

    res = convert_core_to_scmdataframe(
        intermediate,
        period_length=ONE_YEAR_IN_S_INTEGER,
        model="IMAGE",
        scenario="RCP26",
        climate_model="unspecified",
    )

    def round_to_nearest_year(dtin):
        """thank you https://stackoverflow.com/a/48108115"""
        # worth moving to utils?
        dt_start_year = dtin.replace(
            month=1,
            day=1,
            minute=0,
            hour=0,
            second=0,
            microsecond=0
        )
        dt_half_year = dtin.replace(month=6, day=17)
        if dtin > dt_half_year:
            return dt_start_year + relativedelta.relativedelta(years=1)
        else:
            return dt_start_year

    # necessary as moving from even timesteps in seconds does not match perfectly with
    # yearly timesteps (which are not always the same number of seconds apart due to
    # leap years)
    tdata.data.loc[:, "time"] = tdata["time"].apply(round_to_nearest_year)
    res.data.loc[:, "time"] = res["time"].apply(round_to_nearest_year)

    pd.testing.assert_frame_equal(tdata.data, res.data, check_like=True)

import copy
from datetime import datetime, timedelta
from logging import getLogger

import dateutil
import numpy as np
import pandas as pd
from dateutil import parser
from pyam.core import _raise_filter_error, IamDataFrame, read_pandas
from pyam.utils import (
    isstr,
    IAMC_IDX,
    years_match,
    month_match,
    day_match,
    datetime_match,
    hour_match,
    pattern_match,
    to_int,
)

logger = getLogger(__name__)


def read_files(fnames, *args, **kwargs):
    """Read data from a snapshot file saved in the standard IAMC format
    or a table with year/value columns
    """
    if not isstr(fnames):
        raise ValueError(
            "reading multiple files not supported, "
            "please use `openscm.ScmDataFrame.append()`"
        )
    logger.info("Reading `{}`".format(fnames))
    return format_data(read_pandas(fnames, *args, **kwargs))


def format_data(df):
    """Convert an imported dataframe and check all required columns"""
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # all lower case
    str_cols = [c for c in df.columns if isstr(c)]
    df.rename(columns={c: str(c).lower() for c in str_cols}, inplace=True)

    # reset the index if meaningful entries are included there
    if not list(df.index.names) == [None]:
        df.reset_index(inplace=True)

    # format columns to lower-case and check that all required columns exist
    if not set(IAMC_IDX).issubset(set(df.columns)):
        missing = list(set(IAMC_IDX) - set(df.columns))
        raise ValueError("missing required columns `{}`!".format(missing))

    orig = df

    # check whether data in wide format (IAMC) or long format (`value` column)
    if "value" in df.columns:
        # check if time column is given as `year` (int) or `time` (datetime)
        cols = set(df.columns)
        if "year" in cols and "time" not in cols:
            time_col = "year"
        elif "time" in cols and "year" not in cols:
            time_col = "time"
        else:
            msg = "invalid time format, must have either `year` or `time`!"
            raise ValueError(msg)
        extra_cols = list(set(cols) - set(IAMC_IDX + [time_col, "value"]))
        df = df.pivot_table(columns=IAMC_IDX + extra_cols, index=time_col).value
        meta = df.columns.to_frame(index=None)
        df.columns = meta.index
    else:
        # if in wide format, check if columns are years (int) or datetime
        cols = set(df.columns) - set(IAMC_IDX)
        time_cols, extra_cols = [], []
        for i in cols:
            if isinstance(i, (int, float)):
                # a time
                time_cols.append(i)
            else:
                try:
                    d = dateutil.parser.parse(str(i))  # this is datetime
                    time_cols.append(d)
                except ValueError:
                    extra_cols.append(i)  # some other string
        if not time_cols:
            msg = "invalid column format, must contain some time (int, float or datetime) columns!"
            raise ValueError(msg)

        df = df[list(cols - set(extra_cols))].T
        df.index.name = "time"
        meta = orig[IAMC_IDX + extra_cols].set_index(df.columns)

    # cast value columns to numeric, drop NaN's, sort data
    df.dropna(inplace=True, how="all")
    df.sort_index(inplace=True)

    return df, meta


def from_ts(df, index=None, **columns):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if index is not None:
        df.index = index

    # format columns to lower-case and check that all required columns exist
    if not set(IAMC_IDX).issubset(columns.keys()):
        missing = list(set(IAMC_IDX) - set(columns.keys()))
        raise ValueError("missing required columns `{}`!".format(missing))

    df.index.name = "time"

    num_ts = len(df.columns)
    for c_name in columns:
        col = columns[c_name]

        if len(col) == num_ts:
            continue
        if len(col) != 1:
            raise ValueError(
                "Length of column {} is incorrect. It should be length 1 or {}".format(
                    c_name, num_ts
                )
            )
        columns[c_name] = col * num_ts

    meta = pd.DataFrame(columns, index=df.columns)
    return df, meta


def df_append(dfs, inplace=False):
    dfs = [
        df if isinstance(df, ScmDataFrameBase) else ScmDataFrameBase(df) for df in dfs
    ]
    ret = copy.deepcopy(dfs[0]) if not inplace else dfs[0]

    meta = pd.concat([d._meta for d in dfs], ignore_index=True, sort=False)
    data = pd.concat([d._data for d in dfs], axis=1, ignore_index=True, sort=False)
    data.columns = meta.index

    ret._meta = meta
    ret._data = data

    if meta.duplicated().any():
        raise ValueError("Duplicate timeseries")

    if not inplace:
        return ret


class ScmDataFrameBase(object):
    """This base is the class other libraries can subclass

    Having such a subclass avoids a potential circularity where e.g. openscm imports
    ScmDataFrame as well as Pymagicc, but Pymagicc wants to import ScmDataFrame and
    hence to try and import ScmDataFrame you have to import ScmDataFrame itself (hence
    the circularity).
    """

    def __init__(self, data, columns=None, **kwargs):
        """Initialize an instance of an ScmDataFrameBase

        Parameters
        ----------
        data: pd.DataFrame, np.ndarray or string
            A pd.DataFrame or data file with IAMC-format data columns, or a numpy array of timeseries data if `columns` is specified.
            If a string is passed, data will be attempted to be read from file.

        columns: dict
            If None, ScmDataFrameBase will attempt to infer the values from the source.
            Otherwise, use this dict to write the metadata for each timeseries in data. For each metadata key (e.g. "model", "scenario"), an array of values (one per time series) is expected.
            Alternatively, providing an array of length 1 applies the same value to all timeseries in data. For example, if you had three
            timeseries from 'rcp26' for 3 different models 'model', 'model2' and 'model3', the column dict would look like either `col_1` or `col_2`:

            .. code:: python

                >>> col_1 = {
                    "scenario": ["rcp26"],
                    "model": ["model1", "model2", "model3"],
                    "region": ["unspecified"],
                    "variable": ["unspecified"],
                    "unit": ["unspecified"]
                }
                >>> col_2 = {
                    "scenario": ["rcp26", "rcp26", "rcp26"],
                    "model": ["model1", "model2", "model3"],
                    "region": ["unspecified"],
                    "variable": ["unspecified"],
                    "unit": ["unspecified"]
                }
                >>> assert pd.testing.assert_frame_equal(
                    ScmDataFrameBase(d, columns=col_1).meta,
                    ScmDataFrameBase(d, columns=col_2).meta
                )

            Metadata for ['model', 'scenario', 'region', 'variable', 'unit'] is required, otherwise a ValueError will be raised.

        kwargs:
            Additional parameters passed to `pyam.core.read_files` to read nonstandard files
        """
        if columns is not None:
            (_df, _meta) = from_ts(data, **columns)
        elif isinstance(data, IamDataFrame):
            (_df, _meta) = format_data(data.data.copy())
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            (_df, _meta) = format_data(data.copy())
        else:
            (_df, _meta) = read_files(data, **kwargs)
        # force index to be object to avoid unexpected loss of behaviour when
        # pandas can't convert to DateTimeIndex
        _df.index = _df.index.astype("object")
        _df.index.name = "time"
        self._data, self._meta = (_df, _meta)
        self._sort_meta_cols()
        self._format_datetime_col()

    def _sort_meta_cols(self):
        # First columns are from IAMC_IDX and the remainder of the columns are alphabetically sorted
        self._meta = self._meta[
            IAMC_IDX + sorted(list(set(self._meta.columns) - set(IAMC_IDX)))
        ]

    def __len__(self):
        return len(self._meta)

    def to_iamdataframe(self):
        """Convert to  IamDataFrame instance

        Returns
        -------
        An pyam.IamDataFrame instance containing the same data
        """
        return LongIamDataFrame(self.timeseries())

    def to_csv(self, path, **kwargs):
        """Write timeseries data to a csv file

        Parameters
        ----------
        path: string
            file path
        """
        self.to_iamdataframe().to_csv(path, **kwargs)

    def __getitem__(self, key):
        _key_check = [key] if isstr(key) else key
        if key is "time":
            return pd.Series(self._data.index, dtype="object")
        elif key is "year":
            return pd.Series([v.year for v in self._data.index])
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__getitem__(key)
        else:
            return self._data.__getitem__(key)

    def __setitem__(self, key, value):
        _key_check = [key] if isstr(key) else key

        if key is "time":
            self._data.index = pd.Index(value, dtype="object", name="time")
            return value
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__setitem__(key, value)

    def _format_datetime_col(self):
        time_srs = self["time"]

        if isinstance(time_srs.iloc[0], datetime):
            pass
        elif isinstance(time_srs.iloc[0], int):
            self["time"] = [datetime(y, 1, 1) for y in to_int(time_srs)]
        elif isinstance(time_srs.iloc[0], float):

            def convert_float_to_datetime(inp):
                year = int(inp)
                fractional_part = inp - year
                base = datetime(year, 1, 1)
                return base + timedelta(
                    seconds=(base.replace(year=year + 1) - base).total_seconds()
                    * fractional_part
                )

            self["time"] = [convert_float_to_datetime(t) for t in time_srs]

        elif isinstance(self._data.index[0], str):

            def convert_str_to_datetime(inp):
                return parser.parse(inp)

            self["time"] = time_srs.apply(convert_str_to_datetime)

        not_datetime = [not isinstance(x, datetime) for x in self["time"]]
        if any(not_datetime):
            bad_values = self["time"][not_datetime]
            error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(
                bad_values
            )
            raise ValueError(error_msg)

    def timeseries(self, meta=None):
        """Return a pandas dataframe in the same format as pyam.IamDataFrame.timeseries
        Parameters
        ----------
        meta: List of strings
        The list of meta columns that will be included in the rows MultiIndex. If None (default), then all metadata will be used.

        Returns
        -------
        pd.DataFrame with datetimes as columns and each row being a timeseries
        Raises
        ------
        ValueError:
        - if the metadata are not unique between timeseries

        """
        d = self._data.copy()
        meta_subset = self._meta if meta is None else self._meta[meta]
        if meta_subset.duplicated().any():
            raise ValueError("Duplicated meta values")

        d.columns = pd.MultiIndex.from_arrays(
            meta_subset.values.T, names=meta_subset.columns
        )

        return d.T

    @property
    def values(self):
        return self.timeseries().values

    @property
    def meta(self):
        return self._meta.copy()

    def filter(self, keep=True, inplace=False, **kwargs):
        """Return a filtered ScmDataFrame (i.e., a subset of current data)

        Parameters
        ----------
        keep: bool, default True
            keep all scenarios satisfying the filters (if True) or the inverse
        inplace: bool, default False
            if True, do operation inplace and return None
        kwargs:
            The following columns are available for filtering:
             - metadata columns: filter by category assignment
             - 'model', 'scenario', 'region', 'variable', 'unit':
               string or list of strings, where `*` can be used as a wildcard
             - 'level': the maximum "depth" of IAM variables (number of '|')
               (exluding the strings given in the 'variable' argument)
             - 'year': takes an integer, a list of integers or a range
               note that the last year of a range is not included,
               so `range(2010, 2015)` is interpreted as `[2010, ..., 2014]`
             - arguments for filtering by `datetime.datetime`
               ('month', 'hour', 'time')
             - 'regexp=True' disables pseudo-regexp syntax in `pattern_match()`
        """
        _keep_ts, _keep_cols = self._apply_filters(kwargs)
        idx = _keep_ts[:, np.newaxis] & _keep_cols
        assert idx.shape == self._data.shape
        idx = idx if keep else ~idx

        ret = copy.deepcopy(self) if not inplace else self
        d = ret._data.where(idx)
        ret._data = (
            d.dropna(axis=1, how="all").dropna(axis=0, how="all")
        )
        ret._meta = ret._meta[(~d.isna()).sum(axis=0) > 0]

        assert len(ret._data.columns) == len(ret._meta)

        if len(ret._meta) == 0:
            logger.warning("Filtered IamDataFrame is empty!")

        if not inplace:
            return ret

    def _apply_filters(self, filters):
        """Determine rows to keep in data for given set of filters

        Parameters
        ----------
        filters: dict
            dictionary of filters ({col: values}}); uses a pseudo-regexp syntax
            by default, but accepts `regexp: True` to use regexp directly
        """
        regexp = filters.pop("regexp", False)
        keep_ts = np.array([True] * len(self._data))
        keep_col = np.array([True] * len(self.meta))

        # filter by columns and list of values
        for col, values in filters.items():
            if col == "variable":
                level = filters["level"] if "level" in filters else None
                keep_col &= pattern_match(self.meta[col], values, level, regexp).values
            elif col in self.meta.columns:
                keep_col &= pattern_match(self.meta[col], values, regexp=regexp).values
            elif col == "year":
                keep_ts &= years_match(
                    self._data.index.to_series().apply(lambda x: x.year), values
                )

            elif col == "month":
                keep_ts &= month_match(
                    self._data.index.to_series().apply(lambda x: x.month), values
                )

            elif col == "day":
                if isinstance(values, str):
                    wday = True
                elif isinstance(values, list) and isinstance(values[0], str):
                    wday = True
                else:
                    wday = False

                if wday:
                    days = self._data.index.to_series().apply(lambda x: x.weekday())
                else:  # ints or list of ints
                    days = self._data.index.to_series().apply(lambda x: x.day)

                keep_ts &= day_match(days, values)

            elif col == "hour":
                keep_ts &= hour_match(
                    self._data.index.to_series().apply(lambda x: x.hour), values
                )

            elif col == "time":
                keep_ts &= datetime_match(self._data.index, values)

            elif col == "level":
                if "variable" not in filters.keys():
                    keep_col &= pattern_match(
                        self.meta["variable"], "*", values, regexp=regexp
                    ).values
                else:
                    continue

            else:
                _raise_filter_error(col)

        return keep_ts, keep_col

    def head(self, *args, **kwargs):
        return self.timeseries().head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        return self.timeseries().tail(*args, **kwargs)

    def rename(self, mapping, inplace=False):
        """Rename and aggregate column entries using `groupby.sum()` on values.
        When renaming models or scenarios, the uniqueness of the index must be
        maintained, and the function will raise an error otherwise.

        Parameters
        ----------
        mapping: dict
            for each column where entries should be renamed, provide current
            name and target name
            {<column name>: {<current_name_1>: <target_name_1>,
                             <current_name_2>: <target_name_2>}}
        inplace: bool, default False
            if True, do operation inplace and return None
        """
        ret = copy.deepcopy(self) if not inplace else self
        for col, _mapping in mapping.items():
            if col not in self.meta.columns:
                raise ValueError("Renaming by {} not supported!".format(col))
            ret._meta[col] = ret._meta[col].replace(_mapping)
            if ret._meta.duplicated().any():
                raise ValueError("Renaming to non-unique metadata for {}!".format(col))

        if not inplace:
            return ret

    def append(self, other, inplace=False, **kwargs):
        """Appends additional timeseries from a castable object to the current dataframe

        The appending does not occur along a time series

        Parameters
        ----------
        other: openscm.ScmDataFrame or something which can be cast to ScmDataFrameBase
        """
        if not isinstance(other, ScmDataFrameBase):
            other = ScmDataFrameBase(other, **kwargs)

        return df_append([self, other], inplace=inplace)

    def set_meta(self, meta, name=None, index=None):
        """Add metadata columns as pd.Series, list or value (int/float/str)

        Parameters
        ----------
        meta: pd.Series, list, int, float or str
            column to be added to metadata
        name: str, optional
            meta column name (defaults to meta pd.Series.name);
            either a meta.name or the name kwarg must be defined
        """
        # check that name is valid and doesn't conflict with data columns
        if (name or (hasattr(meta, "name") and meta.name)) in [None, False]:
            raise ValueError("Must pass a name or use a named pd.Series")
        name = name or meta.name

        # check if meta has a valid index and use it for further workflow
        if hasattr(meta, "index") and hasattr(meta.index, "names"):
            index = meta.index
        if index is None:
            self._meta[name] = meta
            return

        # turn dataframe to index if index arg is a DataFrame
        if isinstance(index, pd.DataFrame):
            index = index.set_index(
                index.columns.intersection(self._meta.columns).to_list()
            ).index
        if not isinstance(index, (pd.MultiIndex, pd.Index)):
            raise ValueError("index cannot be coerced to pd.MultiIndex")

        meta = pd.Series(meta, index=index, name=name)

        df = self.meta.reset_index()
        if all(index.names):
            df = df.set_index(index.names)
        self._meta = (
            pd.merge(df, meta, left_index=True, right_index=True, how="outer")
            .reset_index()
            .set_index("index")
        )
        self._sort_meta_cols()

    def line_plot(self, x='year', y='value', **kwargs):
        """Helper to generate line plots of timeseries

        See ``pyam.IamDataFrame.line_plot`` for more information

        """
        return self.to_iamdataframe().line_plot(x, y, **kwargs)

    def scatter(self, x, y, **kwargs):
        """Plot a scatter chart using metadata columns

        see pyam.plotting.scatter() for all available options
        """
        self.to_iamdataframe().scatter(x, y, **kwargs)

    def region_plot(self, **kwargs):
        """Plot regional data for a single model, scenario, variable, and year

        see ``pyam.plotting.region_plot()`` for all available options
        """
        return self.to_iamdataframe().region_plot(**kwargs)


class LongIamDataFrame(IamDataFrame):
    """This baseclass is a custom implementation of the IamDataFrame which handles datetime data which spans longer than pd.to_datetime
    can handle
    """

    def _format_datetime_col(self):
        if isinstance(self.data["time"].iloc[0], str):

            def convert_str_to_datetime(inp):
                return parser.parse(inp)

            self.data["time"] = self.data["time"].apply(convert_str_to_datetime)

        not_datetime = [not isinstance(x, datetime) for x in self.data["time"]]
        if any(not_datetime):
            bad_values = self.data[not_datetime]["time"]
            error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(
                bad_values
            )
            raise ValueError(error_msg)

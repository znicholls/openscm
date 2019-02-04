import copy
import os
import warnings
from datetime import datetime
from logging import getLogger

import dateutil
import numpy as np
import pandas as pd
from dateutil import parser
from pyam import IamDataFrame
from pyam.core import _raise_filter_error
from pyam.utils import (
    isstr,
    META_IDX,
    IAMC_IDX,
    years_match,
    month_match,
    day_match,
    datetime_match,
    hour_match,
    pattern_match,
    to_int
)

logger = getLogger(__name__)


def read_pandas(fname, *args, **kwargs):
    """Read a file and return a pd.DataFrame"""
    if not os.path.exists(fname):
        raise ValueError('no data file `{}` found!'.format(fname))
    if fname.endswith('csv'):
        df = pd.read_csv(fname, *args, **kwargs)
    else:
        xl = pd.ExcelFile(fname)
        if len(xl.sheet_names) > 1 and 'sheet_name' not in kwargs:
            kwargs['sheet_name'] = 'data'
        df = pd.read_excel(fname, *args, **kwargs)
    return df


def read_files(fnames, *args, **kwargs):
    """Read data from a snapshot file saved in the standard IAMC format
    or a table with year/value columns
    """
    if not isstr(fnames):
        raise ValueError('reading multiple files not supported, '
                         'please use `pyam.IamDataFrame.append()`')
    logger.info('Reading `{}`'.format(fnames))
    return format_data(read_pandas(fnames, *args, **kwargs))


def format_data(df):
    """Convert an imported dataframe and check all required columns"""
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # all lower case
    str_cols = [c for c in df.columns if isstr(c)]
    df.rename(columns={c: str(c).lower() for c in str_cols}, inplace=True)

    if 'notes' in df.columns:  # this came from the database
        logger.info('Ignoring notes column in dataframe')
        df.drop(columns='notes', inplace=True)
        col = df.columns[0]  # first column has database copyright notice
        df = df[~df[col].str.contains('database', case=False)]
        if 'scenario' in df.columns and 'model' not in df.columns:
            # model and scenario are jammed together in RCP data
            scen = df['scenario']
            df['model'] = scen.apply(lambda s: s.split('-')[0].strip())
            df['scenario'] = scen.apply(
                lambda s: '-'.join(s.split('-')[1:]).strip())

    # reset the index if meaningful entries are included there
    if not list(df.index.names) == [None]:
        df.reset_index(inplace=True)

    # format columns to lower-case and check that all required columns exist
    if not set(IAMC_IDX).issubset(set(df.columns)):
        missing = list(set(IAMC_IDX) - set(df.columns))
        raise ValueError("missing required columns `{}`!".format(missing))

    orig_df = df

    # check whether data in wide format (IAMC) or long format (`value` column)
    if 'value' in df.columns:
        # check if time column is given as `year` (int) or `time` (datetime)
        cols = set(df.columns)
        if 'year' in cols and 'time' not in cols:
            time_col = 'year'
        elif 'time' in cols and 'year' not in cols:
            time_col = 'time'
        else:
            msg = 'invalid time format, must have either `year` or `time`!'
            raise ValueError(msg)
        extra_cols = list(set(cols) - set(IAMC_IDX + [time_col, 'value']))
        df = df.pivot_table(columns=IAMC_IDX + extra_cols, index='year').value
    else:
        # if in wide format, check if columns are years (int) or datetime
        cols = set(df.columns) - set(IAMC_IDX)
        year_cols, time_cols, extra_cols = [], [], []
        for i in cols:
            try:
                int(i)  # this is a year
                year_cols.append(i)
            except (ValueError, TypeError):
                try:
                    dateutil.parser.parse(str(i))  # this is datetime
                    time_cols.append(i)
                except ValueError:
                    extra_cols.append(i)  # some other string
        if year_cols and not time_cols:
            time_col = 'year'
        elif not year_cols and time_cols:
            time_col = 'time'
        else:
            msg = 'invalid column format, must be either years or `datetime`!'
            raise ValueError(msg)

        col_idx = pd.MultiIndex.from_frame(df[IAMC_IDX + extra_cols])
        df = df[cols - set(extra_cols)].T
        df.columns = col_idx

    # cast value columns to numeric, drop NaN's, sort data
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    df.sort_values(META_IDX + ['variable', 'region'] + extra_cols,
                   inplace=True, axis=1)

    return df, time_col, extra_cols, orig_df[set(IAMC_IDX + extra_cols)]


class ScmDataFrameBase(IamDataFrame):
    """This base is the class other libraries can subclass

    Having such a subclass avoids a potential circularity where e.g. openscm imports ScmDataFrame as well as Pymagicc, but Pymagicc wants to import ScmDataFrame and hence to try and import ScmDataFrame you have to import ScmDataFrame itself (hence the circularity).
    """

    def __init__(self, data, **kwargs):
        """Initialize an instance of an ScmDataFrameBase

        Parameters
        ----------
        data: ixmp.TimeSeries, ixmp.Scenario, pd.DataFrame or data file
            an instance of an TimeSeries or Scenario (requires `ixmp`),
            or pd.DataFrame or data file with IAMC-format data columns.
            A pd.DataFrame can have the required data as columns or index.

            Special support is provided for data files downloaded directly from
            IIASA SSP and RCP databases. If you run into any problems loading
            data, please make an issue at:
            https://github.com/IAMconsortium/pyam/issues
        """
        # import data from pd.DataFrame or read from source
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            _data = format_data(data.copy())
        else:
            _data = read_files(data, **kwargs)

        self._data, self.time_col, self.extra_cols, extra_data = _data
        # cast time_col to desired format
        if self.time_col == 'year':
            self._format_year_col()
        elif self.time_col == 'time':
            self._format_datetime_col()

        self._LONG_IDX = IAMC_IDX + [self.time_col] + self.extra_cols

        # define a dataframe for categorization and other metadata indicators
        self.meta = extra_data[META_IDX].drop_duplicates().set_index(META_IDX)
        self.headers = extra_data #
        self.reset_exclude()

        # execute user-defined code
        # if 'exec' in run_control():
        #    self._execute_run_control()

    def __getitem__(self, key):
        _key_check = [key] if isstr(key) else key
        if key is self.time_col:
            return pd.Series(self._data.index)
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__getitem__(key)
        elif set(_key_check).issubset(self.headers.columns):
            return self.headers.__getitem__(key)
        else:
            return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        _key_check = [key] if isstr(key) else key

        if key is self.time_col:
            self._data.index = value
            return value
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__setitem__(key, value)
        else:
            return self.headers.__setitem__(key, value)

    def _format_year_col(self):
        self._data.index = to_int(self._data.index).astype('int')

    def _format_datetime_col(self):
        if isinstance(self._data.index[0], str):
            def convert_str_to_datetime(inp):
                return parser.parse(inp)

            self.index = self.index.apply(convert_str_to_datetime)

        not_datetime = [not isinstance(x, datetime) for x in self._data.index]
        if any(not_datetime):
            bad_values = self.index[not_datetime]
            error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(bad_values)
            raise ValueError(error_msg)

    def timeseries(self, iamc_index=False):
        return self._data.T

    @property
    def data(self):
        df = self._data.T.reset_index()
        dt_cols = set(df.columns) - set(IAMC_IDX) - set(self.extra_cols)
        df = pd.melt(df, id_vars=IAMC_IDX + self.extra_cols, var_name=self.time_col, value_vars=sorted(dt_cols),
                     value_name='value').dropna()
        if self.time_col == 'year':
            df['year'] = df['year'].astype('int64')
        return df

    @data.setter
    def data(self, value):
        self._data, self.time_col, self.extra_cols, extra_data = format_data(value)
        # cast time_col to desired format
        if self.time_col == 'year':
            self._format_year_col()
        elif self.time_col == 'time':
            self._format_datetime_col()

        self._LONG_IDX = IAMC_IDX + [self.time_col] + self.extra_cols

        # define a dataframe for categorization and other metadata indicators
        self.meta = extra_data[META_IDX].drop_duplicates().set_index(META_IDX)

    def filter(self, keep=True, inplace=False, **kwargs):
        """Return a filtered IamDataFrame (i.e., a subset of current data)

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
        ret._data = ret._data.where(idx).dropna(axis=1, how='all').dropna(axis=0, how='all')
        ret.headers = ret.headers[idx.sum(axis=0) > 0]

        idx = pd.MultiIndex.from_tuples(
            pd.unique(list(zip(ret['model'], ret['scenario']))),
            names=('model', 'scenario')
        )
        if len(idx) == 0:
            logger.warning('Filtered IamDataFrame is empty!')

        ret.meta = ret.meta.loc[idx]

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
        regexp = filters.pop('regexp', False)
        keep_ts = np.array([True] * len(self._data))
        keep_col = np.array([True] * len(self.headers))
        headers = self.headers.merge(self.meta, right_index=True, left_on=('model', 'scenario'))

        # filter by columns and list of values
        for col, values in filters.items():
            if col == 'variable':
                level = filters['level'] if 'level' in filters else None
                keep_col &= pattern_match(headers[col], values, level, regexp).values
            elif col in headers.columns:
                keep_col &= pattern_match(headers[col], values, regexp=regexp).values
            elif col == 'year':
                if self.time_col is 'time':
                    keep_ts &= years_match(self._data.index.apply(lambda x: x.year), values)
                else:
                    keep_ts &= years_match(self._data.index, values)

            elif col == 'month':
                if self.time_col is not 'time':
                    _raise_filter_error(col)
                keep_ts &= month_match(self._data.index.apply(lambda x: x.month), values)

            elif col == 'day':
                if self.time_col is not 'time':
                    _raise_filter_error(col)
                if isinstance(values, str):
                    wday = True
                elif isinstance(values, list) and isinstance(values[0], str):
                    wday = True
                else:
                    wday = False

                if wday:
                    days = self._data.index.apply(lambda x: x.weekday())
                else:  # ints or list of ints
                    days = self._data.index.apply(lambda x: x.day)

                keep_ts &= day_match(days, values)

            elif col == 'hour':
                if self.time_col is not 'time':
                    _raise_filter_error(col)
                keep_ts &= hour_match(self._data.index.apply(lambda x: x.hour), values)

            elif col == 'time':
                if self.time_col is not 'time':
                    _raise_filter_error(col)
                keep_ts &= datetime_match(self._data.index, values)

            elif col == 'level':
                if 'variable' not in filters.keys():
                    keep_col &= pattern_match(headers['variable'], '*', values, regexp=regexp).values
                else:
                    continue

            else:
                _raise_filter_error(col)

        return keep_ts, keep_col

    def head(self, *args, **kwargs):
        return self._data.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        return self._data.tail(*args, **kwargs)

    def models(self):
        """Get a list of models"""
        return pd.Series(self.meta.reset_index()['model'].unique(), name='model')

    def scenarios(self):
        """Get a list of scenarios"""
        return pd.Series(self.meta.reset_index()['scenario'].unique(), name='scenario')

    def regions(self):
        """Get a list of regions"""
        return pd.Series(self.headers['region'].unique(), name='region')

    def variables(self, include_units=False):
        """Get a list of variables

        Parameters
        ----------
        include_units: boolean, default False
            include the units
        """
        if include_units:
            return self.headers[['variable', 'unit']].drop_duplicates()
        else:
            return pd.Series(self.headers['variable'].unique(), name='variable')

    def rename(self, mapping, inplace=False):
        res = IamDataFrame(self.data).rename(mapping)

        if inplace:
            self.data = res.data
        else:
            return self.__class__(res.data)

    def require_variable(self, variable, unit=None, year=None,
                         exclude_on_fail=False):
        """Check whether all scenarios have a required variable

        Parameters
        ----------
        variable: str
            required variable
        unit: str, default None
            name of unit (optional)
        year: int or list, default None
            years (optional)
        exclude_on_fail: bool, default False
            flag scenarios missing the required variables as `exclude: True`
        """
        raise NotImplementedError

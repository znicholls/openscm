import os
from datetime import datetime
from logging import getLogger

import dateutil
import pandas as pd
from dateutil import parser
from pyam import IamDataFrame
from pyam.utils import (
    isstr,
    META_IDX,
    IAMC_IDX
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
        cols = df.columns
        if 'year' in cols and 'time' not in cols:
            time_col = 'year'
        elif 'time' in cols and 'year' not in cols:
            time_col = 'time'
        else:
            msg = 'invalid time format, must have either `year` or `time`!'
            raise ValueError(msg)
        extra_cols = list(set(cols) - set(IAMC_IDX + [time_col, 'value']))
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
        df.index.name = time_col
        df.columns = col_idx

    # cast value columns to numeric, drop NaN's, sort data
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

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

        self._data, self.time_col, self.extra_cols, self.meta = _data
        # cast time_col to desired format
        if self.time_col == 'year':
            self._format_year_col()
        elif self.time_col == 'time':
            self._format_datetime_col()

        # self._LONG_IDX = IAMC_IDX + [self.time_col] + self.extra_cols

        # define a dataframe for categorization and other metadata indicators
        self.meta = self.meta.set_index(META_IDX)
        self.reset_exclude()

        # execute user-defined code
        # if 'exec' in run_control():
        #    self._execute_run_control()

    def _format_year_col(self):
        self.index = self._data.index.astype('int')

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
                       value_name='value')
        if self.time_col == 'year':
            df['year'] = df['year'].astype('int64')
        return df

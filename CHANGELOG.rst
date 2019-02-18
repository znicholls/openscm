Changelog
---------

scmcallib
---------

- (`#122 <https://github.com/openclimatedata/openscm/pull/122>`_) Add processing functions ``process_over`` and ``relative_to_ref_period_mean``. Turn off MAGICC adapter tests.
- (`#120 <https://github.com/openclimatedata/openscm/pull/120>`_) Attempt to cast str dates to float as a check for being able to be converted to datetimes
- (`#112 <https://github.com/openclimatedata/openscm/pull/112>`_) Added demo of FaIR adapter
- (`#88 <https://github.com/openclimatedata/openscm/pull/88>`_) Added demos of MAGICC and PH99 adapters
- (`#103 <https://github.com/openclimatedata/openscm/pull/103>`_) Add resample method for `ScmDataFrame`
- (`#101 <https://github.com/openclimatedata/openscm/pull/101>`_) Added PH99 model implementation
- (`#102 <https://github.com/openclimatedata/openscm/pull/102>`_) Added a few plotting routines from pyam to ScmDataFrame for convenience
- (`#100 <https://github.com/openclimatedata/openscm/pull/100>`_) Allow ScmDataFrame to handle data which do not cover the same time span
- (`#99 <https://github.com/openclimatedata/openscm/pull/99>`_) Force ScmDataFrame time to always be a list of datetimes, prevent Pandas coercing to Pandas datetime
- (`#98 <https://github.com/openclimatedata/openscm/pull/98>`_) Allow subsetting the metadata to be included in ``ScmDataFrame.timeseries``
- (`#94 <https://github.com/openclimatedata/openscm/pull/94>`_) Refactor ScmDataFrame to be faster and more specific to SCM data
- (`#89 <https://github.com/openclimatedata/openscm/pull/89>`_) Add ``openscm.scenarios`` module

master
------

- (`#87 <https://github.com/openclimatedata/openscm/pull/87>`_) Added aggregated read of parameter
- (`#86 <https://github.com/openclimatedata/openscm/pull/86>`_) Made top level of region explicit, rather than allowing access via ``()`` and made requests robust to string inputs
- (`#92 <https://github.com/openclimatedata/openscm/pull/92>`_) Updated installation to remove notebook dependencies from minimum requirements as discussed in `#90 <https://github.com/openclimatedata/openscm/issues/90>`_
- (`#85 <https://github.com/openclimatedata/openscm/pull/85>`_) Split out submodule for ScmDataFrameBase ``openscm.scmdataframebase`` to avoid circular imports
- (`#83 <https://github.com/openclimatedata/openscm/pull/83>`_) Rename OpenSCMDataFrame to ScmDataFrame
- (`#78 <https://github.com/openclimatedata/openscm/pull/78>`_) Added OpenSCMDataFrame
- (`#35 <https://github.com/openclimatedata/openscm/pull/35>`_) Add units module
- (`#44 <https://github.com/openclimatedata/openscm/pull/44>`_) Add timeframes module
- (`#40 <https://github.com/openclimatedata/openscm/pull/40>`_) Add parameter handling in core module

import pytest

from openscm.scmdataframe import ScmDataFrame


@pytest.mark.xfail(reason="ScmDataFrame currently unimplemented")
def test_write_read_datafile(test_pd_df, tmp_path):
    tfile = str(tmp_path / "testfile.csv")
    tdf = ScmDataFrame(test_pd_df)

    tdf.to_csv(tfile)

    ScmDataFrame(tfile)

import pytest


from openscm.models import PH99Model


ONE_YEAR = 60*60*24*365.25

@pytest.fixture(scope="function")
def ph_99():
    return PH99Model()

def test_step_time(ph_99):
    ph_99.time = [0, ONE_YEAR]
    ph_99.now = 0

    ph_99.step_time()

    assert ph_99.now == ONE_YEAR

    # TODO: decide about error type
    error_msg = re.escape(
        "Cannot step time again as we are already at the last value in self.time"
    )
    with pytest.raises(AttributeError, match=error_msg):
        ph_99.step_time()

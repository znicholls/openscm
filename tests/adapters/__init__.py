from abc import ABCMeta, abstractmethod
import pytest


@pytest.fixture(scope="function")
def test_adapter(request):
    try:
        yield request.cls.tadapter()
    except TypeError:
        pytest.skip("{} cannot be instantiated".format(str(request.cls.tadapter)))


class AdapterTester():
    @property
    @abstractmethod
    def tadapter(self):
        pass

    @classmethod
    def test_initialize(cls, test_adapter):
        test_adapter.initialize()

    # TODO add standard tests
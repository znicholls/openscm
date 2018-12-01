# skip comparing html output in notebooks as documented on the last page of
# https://media.readthedocs.org/pdf/nbval/latest/nbval.pdf
def pytest_collectstart(collector):
    try:
        collector.skip_compare += "text/html"
    except AttributeError:
        collector.skip_compare = "text/html"


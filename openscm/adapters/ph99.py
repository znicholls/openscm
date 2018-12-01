from ..internal import Adapter
from ..models import PH99Model

"""
Questions as I write:
- how to cache model instances for adapters, doesn't really matter here but matters more models like MAGICC which are expensive to spin up
"""


class PH99(Adapter):
    """Adapter for the simple climate model first presented in Petschel-Held Climatic Change 1999

    This one box model projects global-mean CO2 concentrations, global-mean radiative
    forcing and global-mean temperatures from emissions of CO2 alone.

    Further reference:
    Petschel-Held, G., Schellnhuber, H.-J., Bruckner, T., Toth, F. L., and
    Hasselmann, K.: The tolerable windows approach: Theoretical and methodological
    foundations, Climatic Change, 41, 303–331, 1999.
    """

    def __init__(self):
        self.model = PH99Model()

    # do I need to copy the output or is that inherited from superclass?
    def run(self) -> None:
        # super nice that we don't have to copy docstring from superclass
        self.model.run()

    def step(self) -> None:
        self.model.step()

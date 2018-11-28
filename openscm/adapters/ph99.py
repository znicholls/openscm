from ..internal import Adapter
from ..models import PH99Model

"""
Questions as I write:
- CO2 shortcut?
- How do we specify what mode we want models to run in (emissions driven, concentration driven, inverse from concentrations, inverse from emissions)?
    - Second order question
- variable names, do we copy them straight out of papers or do we use standard coding practices?
    - e.g., in PH99, do we write emissions as E (as in the paper) or e (following Python conventions) or as emissions (giving it an easier name to immediately understand)? An alternate proposal would be to provide access to all variables via the names as they are given in their respective papers, but use Python compliant names internally. This allows external users to query things just by copying the paper, but allows us to write code we can actually work with.
- can/should we cache model instances for models like MAGICC which are expensive to spin up
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


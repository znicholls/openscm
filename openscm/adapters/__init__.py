from .hector import Hector
from .magicc6 import MAGICC6
from .ph99 import PH99
from .fair import FAIR


def get_adapter(model):
    model_classes = {
        "MAGICC6": MAGICC6,
        # "MAGICC7": MAGICC7,
        "Hector": Hector,
        "PH99": PH99,
        "FAIR": FAIR,
    }

    return model_classes[model]

from .hector import Hector
from .magicc6 import MAGICC6

def get_adapter(model):
# TODO: move elsewhere
    model_classes = {
        "MAGICC6": MAGICC6,
        # "MAGICC7": MAGICC7,
        "Hector": Hector,
    }

    return model_classes[model]

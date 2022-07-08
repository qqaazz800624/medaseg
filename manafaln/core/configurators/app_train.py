from collections import OrderedDict
from manafaln.core.configurators import DefaultConfigurator

class TrainConfigurator(DefaultConfigurator):
    def __init__(self, app_name=None, description=None):
        super().__init__(app_name=app_name, description=description)


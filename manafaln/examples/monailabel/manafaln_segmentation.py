from manafaln.adapters.monailabel.configs import ManafalnSegmentation

# Workaround for MONAILabel get_class_names
ManafalnSegmentation.__module__ = __name__

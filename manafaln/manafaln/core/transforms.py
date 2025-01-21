from typing import Dict, List
from monai.transforms import Compose
from manafaln.core.builders import TransformBuilder

def build_transforms(config: List[Dict]) -> Compose:
    """
    Builds a composition of transforms based on the given configuration.

    Args:
        config (List[Dict]): A list of dictionaries representing the configuration for each transform.

    Returns:
        Compose: A composition of transforms.

    """
    builder = TransformBuilder()
    return Compose([builder(c) for c in config])

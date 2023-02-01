from typing import Dict, List
from monai.transforms import Compose
from manafaln.core.builders import TransformBuilder

def build_transforms(config: List[Dict]):
    builder = TransformBuilder()
    return Compose([builder(c) for c in config])

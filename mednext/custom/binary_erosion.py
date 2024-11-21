
import numpy as np
from scipy.ndimage import binary_erosion
from monai.transforms import MapTransform, Transform

class BinaryErosion(Transform):
    def __init__(self, iterations=1, structure=None):
        """
        Apply binary erosion to a single label array.
        :param iterations: Number of erosion iterations.
        :param structure: Structuring element for erosion.
        """
        super().__init__()
        self.iterations = iterations
        self.structure = structure

    def __call__(self, label):
        eroded_label = binary_erosion(label > 0, structure=self.structure, iterations=self.iterations).astype(label.dtype)
        return eroded_label
        
class BinaryErosiond(MapTransform):
    def __init__(self, keys, iterations=1, structure=None):
        """
        Apply binary erosion to specific keys in a dictionary (e.g., label data).
        :param keys: List of keys to apply erosion on.
        :param iterations: Number of erosion iterations.
        :param structure: Structuring element for erosion.
        """
        super().__init__(keys)
        self.iterations = iterations
        self.structure = structure

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = binary_erosion(d[key] > 0, structure=self.structure, iterations=self.iterations).astype(d[key].dtype)
        return d
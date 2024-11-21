#%%

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


#%%

import os, json
import numpy as np
import nibabel as nib
from tqdm import tqdm

data_root = "/data2/open_dataset/MSD/Task09_Spleen"
datalist_path = "/home/u/qqaazz800624/manafaln/mednext/datalist/datalist_spleen.json"
with open(datalist_path, "r") as f:
    datalist = json.load(f)


folds = ['training', 'validation', 'testing']
save_root = "/neodata/hap/Task09_Spleen/labelsTr"
erosion_transform = BinaryErosion(iterations=2)

for fold in folds:
    for i in tqdm(range(len(datalist[fold]))):
        label_path = os.path.join(data_root, datalist[fold][i]['label'])
        basename = os.path.basename(label_path)
        label_nifti = nib.load(label_path)
        label_data = label_nifti.get_fdata().astype(np.uint8)
        
        eroded_label_data = erosion_transform(label_data)
        eroded_label_nifti = nib.Nifti1Image(eroded_label_data, affine=label_nifti.affine)
        
        nib.save(eroded_label_nifti, os.path.join(save_root, basename))
    

#%%






#%%





#%%






#%%
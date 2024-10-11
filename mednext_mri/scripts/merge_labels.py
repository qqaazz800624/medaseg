#%%
import json
from pathlib import Path
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

mapping = {
    0: 0,    # background
    2: 1,    # kidney
    3: 1,    # kidney
    6: 2,    # liver
    10: 3,   # pancreas
    1: 4     # spleen
}

for i in range(30):
    mapping.setdefault(i, 0)

mapping.setdefault(i, 0)

data_root = '/neodata/hap/AMOS_MRI'
data_list = 'datalist_mr.json' 
data_path = os.path.join(data_root, data_list)

with open(data_path, "r") as f:
    dataset = json.load(f)

#%%

dataroot = Path("/neodata/hap/AMOS_MRI")
datalist = {}

for split in ["training", "validation"]:
    datalist[split] = []
    for case in tqdm(dataset[split]):
        seg_name = Path(case["label"]).name
        
        # Convert label format
        seg_nifti = nib.load(str(dataroot / case["label"]))
        seg_data = np.asanyarray(seg_nifti.dataobj)
        seg_data = np.vectorize(mapping.get)(seg_data)
        seg_nifti = nib.Nifti1Image(seg_data.astype(np.uint8), seg_nifti.affine)
        if split == "training":
            nib.save(seg_nifti, f"/neodata/hap/AMOS_MRI/labelsTr/{seg_name}")
        else:
            nib.save(seg_nifti, f"/neodata/hap/AMOS_MRI/labelsVa/{seg_name}")

#%%

import nibabel as nib
import numpy as np
import os

data_root = '/neodata/hap/AMOS_MRI'
#file_name = 'labelsTr/amos_0507.nii.gz'
file_name = 'labelsVa/amos_0544.nii.gz'

file_path = os.path.join(data_root, file_name)

nii = nib.load(file_path)
nii_data = nii.get_fdata()

#%%

nii_data.shape
#%%

nii_data

#%%

np.unique(nii_data)


#%%

import json
json_path = '/home/u/qqaazz800624/manafaln/mednext_mri/AMOS22/dataset.json'

with open(json_path) as f:
    data = json.load(f)

data


#%%








#%%
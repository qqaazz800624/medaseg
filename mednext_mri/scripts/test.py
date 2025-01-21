#%%

import os
import nibabel as nib
import json
import numpy as np

#%%

datalist_path = "/home/u/qqaazz800624/manafaln/mednext_mri/datalist/datalist_mr.json"

with open(datalist_path, "r") as f:
    datalist = json.load(f)

datalist.keys()

for key in datalist.keys():
    print(key, len(datalist[key]))

#%%

data_root = "/home/u/qqaazz800624/manafaln/mednext_mri/scripts/amos22/labels"

label_path = os.path.join(data_root, 'LB_AMOS_0001.nii.gz')
label = nib.load(label_path)
label_data = label.get_fdata()
label_data = np.asanyarray(label_data)
label_data




#%%

np.unique(label_data)






#%%
#%%

import json
import os


root_dir = '/home/u/qqaazz800624/manafaln/mednext_mri'
datalist_ct_mr_path = os.path.join(root_dir, 'datalist/datalist_ct_mr.json')
datalist_ct_path = os.path.join(root_dir, 'datalist/datalist_ct.json')


with open(datalist_ct_mr_path, 'r') as f:
    datalist_ct_mr = json.load(f)

with open(datalist_ct_path, 'r') as f:
    datalist_ct = json.load(f)

#%%
# Remove MR images that have corresponding CT images in the training and validation sets
datalist_mr = {}
data_splits = ['training', 'validation']

for split in data_splits:
    ct_image_label_pairs = set((item['image'], item['label']) for item in datalist_ct[split])

    filtered_pairs = [
        item for item in datalist_ct_mr[split]
        if (item['image'], item['label']) not in ct_image_label_pairs
    ]

    datalist_mr[split] = filtered_pairs


#%%
# Remove MR images that have corresponding CT images in the testing set
ct_image_paths = set(item['image'] for item in datalist_ct['testing'])
filtered = [item for item in datalist_ct_mr['testing'] if item['image'] not in ct_image_paths]
datalist_mr['testing'] = filtered

#%%

with open(os.path.join(root_dir, 'datalist/datalist_mr.json'), 'w') as f:
    json.dump(datalist_mr, f, indent=4)


#%%









#%%










#%%
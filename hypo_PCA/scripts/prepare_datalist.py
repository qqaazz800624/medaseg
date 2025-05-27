#%%
import os
import json
from sklearn.model_selection import KFold

#%%

# Path to data folder
data_root = "/neodata/open_dataset/PINKCC"

# Subfolders for CT and segmentation
ct_folders = ["MSKCC", "TCGA"]

# Prepare data pairs
data_pairs = []
for folder in ct_folders:
    ct_path = os.path.join(data_root, "CT", folder)
    seg_path = os.path.join(data_root, "Segmentation", folder)

    ct_files = sorted(os.listdir(ct_path))

    for file in ct_files:
        ct_file_path = f"CT/{folder}/{file}"
        seg_file_path = f"Segmentation/{folder}/{file}"
        data_pairs.append({"image": ct_file_path, "label": seg_file_path})

# Split data into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

folds = {}
for fold_index, (_, test_indices) in enumerate(kf.split(data_pairs)):
    fold_key = f"fold_{fold_index}"
    folds[fold_key] = [data_pairs[idx] for idx in test_indices]

# Save as JSON
output_json = "datalist.json"
with open(output_json, "w") as json_file:
    json.dump(folds, json_file, indent=4)

print(f"JSON datalist saved to {output_json}")
# %%

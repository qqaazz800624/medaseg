#%%

import os
import json

# Paths
data_root = "/neodata/open_dataset/PINKCC"
eval_folder = os.path.join(data_root, "PINKCC_phase1_data")
original_datalist_path = os.path.join(data_root, "datalist.json")
new_datalist_path = os.path.join(data_root, "datalist_eval.json")

# Step 1: Load original datalist
with open(original_datalist_path, "r") as f:
    datalist = json.load(f)

# Step 2: Prepare fold_eval entries (images only, no labels)
eval_files = sorted([
    f for f in os.listdir(eval_folder) if f.endswith(".nii.gz")
])

#%%

fold_eval = [{"image": f"PINKCC_phase1_data/{fname}"} for fname in eval_files]
datalist['fold_eval'] = fold_eval

#%%

with open(new_datalist_path, "w") as f:
    json.dump(datalist, f, indent=4)

print(f"Updated datalist with fold_eval saved to {new_datalist_path}")



# %%

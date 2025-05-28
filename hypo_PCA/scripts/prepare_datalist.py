#%%
import os
import json
from sklearn.model_selection import KFold

#%%

# Dataset root
data_root = "/neodata/hsu/hypo/Final_dataset/HYPO_NTUH_dataset"
output_json = "datalist.json"
excluded_patient = "3729053"

# patient_ids = sorted([
#     name for name in os.listdir(data_root)
#     if os.path.isdir(os.path.join(data_root, name))
# ])

patient_ids = sorted([
    pid for pid in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, pid)) and pid != excluded_patient
])

#%%

# Build list of image-label dictionaries
samples = []
for pid in patient_ids:
    sample = {
        "image": [
            f"{pid}/Bspline_6DOF_regi_SITK_corrected_T1c_origin.nii.gz",
            f"{pid}/Bspline_6DOF_regi_SITK_corrected_T1n_origin.nii.gz",
            f"{pid}/SITK_corrected_T2_origin.nii.gz"
        ],
        "label": f"{pid}/T2_label.nii.gz"
    }
    samples.append(sample)

# Prepare 10-fold split
kf = KFold(n_splits=10, shuffle=True, random_state=42)
folds = {}

for fold_idx, (_, test_idx) in enumerate(kf.split(samples)):
    fold_key = f"fold_{fold_idx}"
    folds[fold_key] = [samples[i] for i in test_idx]


#%%
# Save JSON
with open(os.path.join(data_root, output_json), "w") as f:
    json.dump(folds, f, indent=4)

# with open(output_json, "w") as f:
#     json.dump(folds, f, indent=4)

print(f"10-fold datalist saved to {os.path.join(data_root, output_json)}")

# %%

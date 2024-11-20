#%%
import json
import os

root_dir = "/home/u/qqaazz800624/manafaln/mednext/datalist"
json_name = "datalist.json"
# Load the original datalist
with open(os.path.join(root_dir, json_name), 'r') as file:
    datalist = json.load(file)


#%%
# Function to update paths
def update_paths(datalist):
    for split in ["training", "validation", "testing"]:
        for entry in datalist[split]:
            # Extract the numerical part from the file name and build the new path
            image_id = entry["image"].split('_')[-1].split('.')[0]  # Extract '2' from 'IM_SPL_2.nii.gz'
            entry["image"] = f"imagesTr/spleen_{image_id}.nii.gz"
            entry["label"] = f"labelsTr/spleen_{image_id}.nii.gz"
    return datalist

# Update the paths
updated_datalist = update_paths(datalist)

# Save the updated datalist to a new JSON file
with open(os.path.join(root_dir, "datalist_spleen.json"), 'w') as file:
    json.dump(updated_datalist, file, indent=4)

print("Paths updated and saved to 'updated_datalist.json'")
#%%
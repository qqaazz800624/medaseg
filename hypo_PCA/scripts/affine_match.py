#%%

# import os
# import json
# from monai.transforms import LoadImage, ResampleToMatch

# data_root = "/neodata/hsu/hypo/Final_dataset/HYPO_NTUH_dataset_affine_matched"
# output_json = "datalist.json"

# with open(os.path.join(data_root, output_json), "r") as f:
#     datalist = json.load(f)

# datalist


# #%%

# image_loader = LoadImage(image_only=False)

# datalist["fold_0"][0]["image"][0]
# image1, meta1 = image_loader(os.path.join(data_root, datalist["fold_0"][0]["image"][0]))
# image2, meta2 = image_loader(os.path.join(data_root, datalist["fold_0"][0]["image"][1]))
# image3, meta3 = image_loader(os.path.join(data_root, datalist["fold_0"][0]["image"][2]))
# label, meta_label = image_loader(os.path.join(data_root, datalist["fold_0"][0]["label"]))




# #%%


# print('image1 affine:', meta1['affine'])
# print('image2 affine:', meta2['affine'])
# print('image3 affine:', meta3['affine'])
# print('label affine:', meta_label['affine'])





#%%

import os
import json
import torch
import numpy as np
import SimpleITK as sitk
from monai.transforms import LoadImage, ResampleToMatch
from tqdm import tqdm

data_root = "/neodata/hsu/hypo/Final_dataset/HYPO_NTUH_dataset"
datalist_path = os.path.join(data_root, "datalist.json")
output_root = "/neodata/hsu/hypo/Final_dataset/HYPO_NTUH_dataset_affine_matched"
os.makedirs(output_root, exist_ok=True)

with open(datalist_path, "r") as f:
    datalist = json.load(f)

image_loader = LoadImage(image_only=False)
resampler = ResampleToMatch()

for fold_name in datalist:
    for idx, case in enumerate(tqdm(datalist[fold_name], desc=f"Processing {fold_name}")):
        img_paths = [os.path.join(data_root, p) for p in case["image"]]
        label_path = os.path.join(data_root, case["label"])
        
        # 讀取 label
        label, meta_label = image_loader(label_path)
        label_np = np.asanyarray(label)
        label_t = torch.as_tensor(label_np).unsqueeze(0)

        # 儲存新資料夾
        pid = case["image"][0].split('/')[0]
        out_dir = os.path.join(output_root, pid)
        os.makedirs(out_dir, exist_ok=True)

        # 對每個 image 做對齊 & 存檔
        for i, img_path in enumerate(img_paths):
            img, meta = image_loader(img_path)
            img_np = np.asanyarray(img)
            img_t = torch.as_tensor(img_np).unsqueeze(0)
            # 對齊
            img_aligned = resampler(img_t, label_t, mode="bilinear")
            # 去除 channel 維 (1, H, W, D) → (H, W, D)
            img_aligned_np = img_aligned.squeeze(0).cpu().numpy()
            # 轉成 (z, y, x) for SimpleITK
            img_aligned_np_sitk = img_aligned_np.transpose(2, 1, 0)
            img_sitk = sitk.GetImageFromArray(img_aligned_np_sitk)
            label_sitk = sitk.ReadImage(label_path)
            img_sitk.CopyInformation(label_sitk)
            # 新檔案名稱
            out_img_name = os.path.basename(img_path)
            out_img_path = os.path.join(out_dir, out_img_name)
            sitk.WriteImage(img_sitk, out_img_path)

        # label 也一起 copy 一份（也做 axis 轉換）
        label_np_sitk = label_np.transpose(2, 1, 0)
        label_sitk_new = sitk.GetImageFromArray(label_np_sitk)
        label_sitk_new.CopyInformation(label_sitk)
        out_label_name = os.path.basename(label_path)
        out_label_path = os.path.join(out_dir, out_label_name)
        sitk.WriteImage(label_sitk_new, out_label_path)





#%%
#%%
import json
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

tqdm.pandas()

def read_csv(df_path, dir):
    print("Reading", df_path)
    # Read raw csv file
    df = pd.read_csv(df_path, index_col=0)

    # Fill NA with None
    df = df.where(pd.notnull(df), None)

    # Prepare data to save in datalist
    def prepare(row):
        uid = row.name
        row["uid"] = uid
        row["img"] = os.path.join(dir, "dicom", f"{uid}.dcm")
        row["seg"] = os.path.join(dir, "annotation", f"{uid}.json")
        row["clf_ett"] = [row["ETT_NA"], row["ETT_Normal"], row["ETT_Abnormal"]]
        row["clf_ngt"] = [row["NGT_NA"], row["NGT_Normal"], row["NGT_Abnormal"]]
        return row

    df = df.progress_apply(prepare, axis=1)
    return df

def read_clip_csv(df_path, dir):
    print("Reading", df_path)
    # Read raw csv file
    df = pd.read_csv(df_path, index_col=0)

    # Fill NA with None
    df = df.where(pd.notnull(df), None)

    # Prepare data to save in datalist
    def prepare(row):
        uid = row.name
        row["uid"] = uid
        row["img"] = os.path.join(dir, "train", f"{uid}.jpg")
        seg_path = os.path.join(dir, "annotations", f"{uid}.json")
        row["seg"] = seg_path if os.path.exists(seg_path) else None
        row["clf_ett"] = [row["ETT_NA"], row["ETT_Normal"], row["ETT_Abnormal"]]
        row["clf_ngt"] = [row["NGT_NA"], row["NGT_Normal"], row["NGT_Abnormal"]]
        return row

    df = df.progress_apply(prepare, axis=1)
    return df

def df_to_datalist(df):
    df = df[["uid", "img", "seg", "clf_ett", "clf_ngt"]]
    return df.to_dict(orient="records")

def split_k_fold(df, k=5):
    def get_state(row):
        state = 0
        for k, v in row.items():
            if ("clf" in k) and (v[-1] == 1):
                state = 1
        return state
    k_fold_datalist = {}
    # Stratified splitting
    for i, (_, fold) in enumerate(StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(df, df.apply(get_state, axis=1))):
        k_fold_datalist[f"fold_{i}"] = df_to_datalist(df.iloc[fold])
    return k_fold_datalist

def save_datalist(datalist, path):
    # Save datalist to json
    with open(path, 'w') as fp:
        json.dump(datalist, fp, indent=4)

if __name__ == "__main__":
    df_train = read_csv(df_path="/data2/smarted/PXR/data/P5767_Preprocessed/label.csv", dir="/data2/smarted/PXR/data/P5767_Preprocessed")
    datalist = split_k_fold(df_train, k=5)

    df_test = read_csv(df_path="/data2/smarted/PXR/data/C426_Catheter/label.csv", dir="/data2/smarted/PXR/data/C426_Catheter")
    datalist["NTUH-20"] = df_to_datalist(df_test[df_test["Group"] == "C4"])
    datalist["NTUH-YB"] = df_to_datalist(df_test[df_test["Group"] == "C8"])

    df_clip = read_clip_csv(df_path="/data2/smarted/PXR/data/CLiP/label.csv", dir="/data2/smarted/PXR/data/CLiP")
    datalist["CLiP_with_seg"] = df_to_datalist(df_clip[df_clip["seg"].apply(lambda x: x is None)])
    datalist["CLiP_without_seg"] = df_to_datalist(df_clip[df_clip["seg"].apply(lambda x: x is not None)])

    save_datalist(datalist, "data/catheter_datalist.json")

# %%

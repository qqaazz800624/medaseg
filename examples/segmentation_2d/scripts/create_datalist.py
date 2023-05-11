import json
import os

from sklearn.model_selection import train_test_split


def create_datalist(uids, image_dir, label_dir=None):
    datalist = []
    for uid in uids:
        data = {}
        data["image"] = os.path.join(image_dir, uid)
        if label_dir is not None:
            data["label"] = os.path.join(label_dir, uid)
        datalist.append(data)
    return datalist


def main(
    train_valid_uids,
    test_uids,
    train_valid_image_dir,
    train_valid_label_dir,
    test_image_dir,
    datalist_save_path,
):
    train_uids, valid_uids = train_test_split(
        train_valid_uids, test_size=0.2, random_state=42
    )

    datalist = {}
    datalist["train"] = create_datalist(
        uids=train_uids,
        image_dir=train_valid_image_dir,
        label_dir=train_valid_label_dir,
    )
    datalist["valid"] = create_datalist(
        uids=valid_uids,
        image_dir=train_valid_image_dir,
        label_dir=train_valid_label_dir,
    )
    datalist["test"] = create_datalist(
        uids=test_uids,
        image_dir=test_image_dir,
    )

    os.makedirs(os.path.dirname(datalist_save_path), exist_ok=True)
    with open(datalist_save_path, "w") as f:
        json.dump(datalist, f, indent=4)


if __name__ == "__main__":
    root = "data"
    train_valid_image_dir = "train_2d/images"
    train_valid_label_dir = "train_2d/labels"
    test_image_dir = "test_2d/images"
    datalist_save_path = "data/datalist.json"

    train_valid_uids = os.listdir(os.path.join(root, train_valid_image_dir))
    test_uids = os.listdir(os.path.join(root, test_image_dir))

    main(
        train_valid_uids=train_valid_uids,
        test_uids=test_uids,
        train_valid_image_dir=train_valid_image_dir,
        train_valid_label_dir=train_valid_label_dir,
        test_image_dir=test_image_dir,
        datalist_save_path=datalist_save_path,
    )

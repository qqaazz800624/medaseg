# 2D Liver Segmentation Example

## About The Project

This code example demonstrates 2D segmentation of the liver in CT scans using Manafaln.

## Table of Contents

- [2D Liver Segmentation Example](#2d-liver-segmentation-example)
  - [About The Project](#about-the-project)
  - [Table of Contents](#table-of-contents)
  - [Built with](#built-with)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setting Up the Workspace](#setting-up-the-workspace)
  - [Setting Up the Environment](#setting-up-the-environment)
- [Data](#data)
  - [Downloading the Dataset](#downloading-the-dataset)
  - [Creating the Data List](#creating-the-data-list)
- [Training](#training)
  - [Setting Up the Configurations](#setting-up-the-configurations)
    - [Trainer Configuration](#trainer-configuration)
    - [Workflow Configuration](#workflow-configuration)
    - [Data Configuration](#data-configuration)
  - [Starting the Training](#starting-the-training)
- [Testing](#testing)
  - [Setting Up the Configurations](#setting-up-the-configurations-1)
  - [Starting the Testing](#starting-the-testing)
- [Inference](#inference)
  - [Setting Up the Configurations](#setting-up-the-configurations-2)
  - [Starting the Inference](#starting-the-inference)

## Built with

* [MONAI](https://monai.io/) - A PyTorch-based, open-source framework for deep learning in medical imaging.
* [Lightning](https://www.lightning.ai/) - A PyTorch-based, open-source framework for deep learning research that aims to standardize the implementation of common research tasks.
* [Manafaln](https://gitlab.com/nanaha1003/manafaln) - A package developed by our lab that provides a highly flexible configuration-based training tool for model training with MONAI and Lightning frameworks in medical imaging.

# Installation

## Requirements

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) version 22.11.1 or later.

## Setting Up the Workspace

1. Create an empty project workspace by running the following command in your terminal:
    ```sh
    mkdir liver-seg-2d
    cd liver-seg-2d
    ```

2. Clone the `Manafaln` package by running the following command in your terminal:
    ```sh
    git clone https://gitlab.com/nanaha1003/manafaln.git
    ```

3. Copy the `segmentation_2d` project from the `examples` directory into your project workspace by running the following command:
    ```sh
    cp -r manafaln/examples/segmentation_2d/* .
    ```

4. After copying the files, your project workspace should have the following folder structure:
    ```
    liver-seg-2d
    ├── commands
    ├── configs
    ├── custom
    ├── manafaln
    ├── scripts
    ├── README.md
    └── requirements.txt
    ```

This folder structure is a standard layout for a `Manafaln` project, with each folder serving a specific purpose:

- `commands`: This folder contains shell scripts for executing specific tasks, such as model training or testing.
- `configs`: This folder contains YAML configuration files for the training, testing, and inference processes.
- `custom`: This folder contains custom modules that can be easily imported by specifying the `path` in the configuration file.
- `data`: This folder contains the project's data.
- `lightning_logs`: This folder stores training logs and model checkpoints.
- `manafaln`: This folder contains the source code for the `Manafaln` package.
- `scripts`: This folder contains miscellaneous Python scripts for data processing, evaluation, or visualization.
- `README.md`: A file that explains the purpose of the project, provides installation instructions, and outlines usage guidelines.
- `requirements.txt`: A file containing a list of required Python packages and their versions. Package managers like `pip` use this file to install the necessary dependencies.

## Setting Up the Environment

1. Create a new `conda` environment by running the following commands in your terminal:
    ```sh
    conda create -n liver-seg-2d python==3.9.12
    conda activate liver-seg-2d
    ```
    This creates a new environment called liver-seg-2d with Python version 3.9.12 and activates it.

2. Install the `Manafaln` and required packages by running the following command in your terminal:
    ```sh
    pip install manafaln/
    pip install -r requirements.txt
    ```
    This installs the required packages into your current environment.

# Data

## Downloading the Dataset

1. Obtain a Kaggle API Token by signing up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username\>/account) and select 'Create New Token'.
This will trigger the download of `kaggle.json`, a file containing your API credentials.

2. Export your Kaggle username and token to the environment by running the following commands in your terminal:
    ```sh
    export KAGGLE_USERNAME=<username>
    export KAGGLE_KEY=<key>
    ```
    \<username\> and \<key\> are the credentials found in the downloaded `kaggle.json`.

3. Install the Kaggle API by running the following command in your terminal:
    ```sh
    pip install kaggle
    ```

4. Download the data from the [MeDA Lab Internal Competition | Kaggle](https://www.kaggle.com/competitions/meda-lab-internal-competition/data) by running the following command in your terminal:
    ```sh
    kaggle competitions download -c meda-lab-internal-competition -p data
    ```
    This downloads the competition data to the data folder within your project directory.

5. Unzip the downloaded data by running the following command in your terminal:
    ```sh
    unzip data/meda-lab-internal-competition.zip -d data
    ```
    This extracts the downloaded data to the data folder within your project directory.

6. Once done, you should see the following folder structure
    ```
    liver-seg-2d
    └── data
        ├── train_2d
        │   ├── images
        │   └── labels
        └── test_2d
            └── images
    ```

## Creating the Data List

1. Create a data list by running the following command in your terminal:
    ```sh
    python scripts/create_datalist.py
    ```
    This creates a `datalist.json` file in the data folder, which contains the file paths for the training and validation images and labels.

2. Once done, you should see the `datalist.json` file in the `data` folder. The sample layout is as follows:

    <details>
    <summary>Example</summary>

    ```json
    {
        "train": [
            {
                "image": "train_2d/images/liver_xx_xx.nii",
                "label": "train_2d/labels/liver_xx_xx.nii"
            },
            ...
        ],
        "valid": [
            {
                "image": "train_2d/images/liver_xx_xx.nii",
                "label": "train_2d/labels/liver_xx_xx.nii"
            },
            ...
        ],
        "test":[
            {
                "image": "test_2d/images/liver_xx_xx.nii"
            },
            ...
        ]
    }
    ```

    </details>
    <br>

# Training

## Setting Up the Configurations

The training configuration file is located at `configs/train.yaml` and contains all the necessary information for the current training. There are three main components in the configuration file: `trainer`, `data`, and `workflow`. The sample layout of the configuration file is as follows:

```yaml
trainer:
    ...

workflow:
    ...

data:
    ...
```

A module is defined with three properties: `name`, `path`, and `args`.

- `name`: Specifies the name of the module.
- `path`: Specifies the path to import the module.
- `args`: Keywords arguments that are used to initialize the instance.

Here is an example layout of a module in YAML format:

```yaml
name: module_name
path: path.to.module
args:
  arg1: value1
  arg2: value2
```

### Trainer Configuration

The `trainer` section contains two parts: `settings` and `callbacks`. The options in settings will be directly use to construct `pytorch_lightning.Trainer`, you can find all options from [PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html). The `callbacks` are the PyTorch Lightning callback objects for the trainer.

Note that "&" is a reference in YAML that can be reference later with "*".

<details>
<summary>Example</summary>

```yaml
trainer:
  settings:
    accelerator: gpu
    devices: [0]

    max_epochs: &max_epochs 10

  callbacks:
    - name: ModelCheckpoint
      args:
        filename: best_model
        monitor: val_meandice
        mode: max
```

</details>
<br>

### Workflow Configuration

The `workflow` section contains all informations needed to build a PyTorch Lightning `LightningModule`.  This is where we define `model`, `loss`, `metrics`, `optimizer`, `scheduler`, and `post_transforms`.

Note that "*" is used to reference "&" that we defined earlier.

<details>
<summary>Example</summary>

```yaml
workflow:
  name: SupervisedLearningV2

  settings:
    scheduler:
      interval: epoch
      frequency: 1

  components:
    model:
      name: UNet
      args:
        spatial_dims: 2
        in_channels: 1
        out_channels: 1
        channels: [4, 8, 16]
        strides: [2, 2]

    loss:
    - name: DiceLoss
      input_keys: [preds, label]
      args:
        sigmoid: True

    optimizer:
      name: AdamW
      args:
        lr: 3.0e-4
        weight_decay: 1.0e-5

    scheduler:
      name: CosineAnnealingWarmRestarts
      args:
        T_0: *max_epochs

    post_transforms:
      validation:
      - name: Activationsd
        args:
          keys: preds
          sigmoid: True
      - name: AsDiscreted
        args:
          keys: preds
          threshold: 0.5

    metrics:
      validation:
      - name: MONAIAdapter
        log_label: val_meandice
        args:
          name: DiceMetric
```

</details>
<br>

### Data Configuration

The `data` section contains all informations needed to build a PyTorch Lightning `LightningDataModule`. This is where we define `transforms`, `dataset`, and `dataloader` for `training` and `validation`.

<details>
<summary>Example</summary>

```yaml
data:
  name: DecathlonDataModule

  settings:
    data_root: data
    data_list: data/datalist.json

  training:
    data_list_key: train
    transforms:
    - name: LoadImaged
      args:
        keys: [image, label]
    - name: RandAffined
      args:
        keys: [image, label]
        prob: 1.0
        rotate_range: 0.25
        shear_range: 0.2
        translate_range: 0.1
        scale_range: 0.2
        padding_mode: zeros
    - name: EnsureTyped
      args:
        keys: [image, label]
        dtype: float32
        track_meta: False

    dataset:
      name: Dataset

    dataloader:
      name: DataLoader
      args:
        batch_size: 8
        shuffle: True
        num_workers: 4

  validation:
    data_list_key: valid
    transforms:
    - name: LoadImaged
      args:
        keys: [image, label]
        image_only: True
    - name: EnsureTyped
      args:
        keys: [image, label]
        dtype: float32
        track_meta: False

    dataset:
      name: Dataset

    dataloader:
      name: DataLoader
      args:
        batch_size: 20
        num_workers: 4
```

</details>
<br>

## Starting the Training
1. Run the training command by executing the following shell script:
    ```sh
    bash commands/train.sh
    ```

2. The training logs and model checkpoints can be found in the `lightning_logs` directory.

3. (Optional) Monitor the training logs on Tensorboard by running the following command:
    ```sh
    tensorboard --logdir lightning_logs/ --bind_all --port <port>
    ```
    Replace <port> with the desired port number.

# Testing
## Setting Up the Configurations

The testing configuration file is `configs/test.yaml`. The file structure is similar to the training configuration file, but only `validation`-related fields are used.

<details>
<summary>Example</summary>

```yaml
trainer:
  settings:
    accelerator: gpu
    devices: [0]

workflow:
  name: SupervisedLearningV2

  components:
    model:
      name: UNet
      args:
        spatial_dims: 2
        in_channels: 1
        out_channels: 1
        channels: [4, 8, 16]
        strides: [2, 2]

    post_transforms:
      validation:
      - name: Activationsd
        args:
          keys: preds
          sigmoid: True
      - name: AsDiscreted
        args:
          keys: preds
          threshold: 0.5

    metrics:
      validation:
      - name: MONAIAdapter
        log_label: val_meandice
        args:
          name: DiceMetric

data:
  name: DecathlonDataModule

  settings:
    data_root: data
    data_list: data/datalist.json

  validation:
    data_list_key: valid
    transforms:
    - name: LoadImaged
      args:
        keys: [image, label]
        image_only: True
    - name: EnsureTyped
      args:
        keys: [image, label]
        dtype: float32
        track_meta: False

    dataset:
      name: Dataset

    dataloader:
      name: DataLoader
      args:
        batch_size: 20
        num_workers: 4
```

</details>
<br>

## Starting the Testing

1. Run the testing command by executing the following shell script, where <version_no> is the log version in `lightning_logs/version_<version_no>` corresponding to the trained model you want to test:
    ```sh
    bash commands/test.sh <version_no>
    ```

2. The testing result will be printed in the terminal output in the following format:
    ```
    val_meandice          0.xxxxx
    ```
    Here, `val_meandice` is the name of the metric being evaluated and 0.xxxxx is the value of the metric on the testing dataset.

# Inference

## Setting Up the Configurations

The inference configuration file is `configs/inference.yaml`. The file structure is similar to training configuration file, but only `predict`-related fields are used.

<details>
<summary>Example</summary>

```yaml
trainer:
  settings:
    accelerator: gpu
    devices: [0]

workflow:
  name: SupervisedLearningV2

  settings:
    decollate:
      predict:
      - image
      - image_meta_dict
      - preds

  components:
    model:
      name: UNet
      args:
        spatial_dims: 2
        in_channels: 1
        out_channels: 1
        channels: [4, 8, 16]
        strides: [2, 2]

    post_transforms:
      predict:
      - name: Activationsd
        args:
          keys: preds
          sigmoid: True
      - name: AsDiscreted
        args:
          keys: preds
          threshold: 0.5
      - name: SaveRunLengthEncodingd
        path: custom.run_length_encoder
        args:
          keys: preds
          meta_keys: image_meta_dict
          output_dir: data
          filename: predictions.csv

data:
  name: DecathlonDataModule

  settings:
    data_root: data
    data_list: data/datalist.json

  predict:
    data_list_key: test
    transforms:
    - name: LoadImaged
      args:
        keys: image
    - name: EnsureTyped
      args:
        keys: image
        dtype: float32
        track_meta: False

    dataset:
      name: Dataset

    dataloader:
      name: DataLoader
      args:
        batch_size: 1
        pin_memory: False
        num_workers: 8
```

</details>
<br>

## Starting the Inference

1. Run the inference command by executing the following shell script, where <version_no> is the log version in `lightning_logs/version_<version_no>` corresponding to the trained model you want to use for inference:
    ```sh
    bash commands/inference.sh <version_no>
    ```

2. The inference results will be saved as CSV file at `data/data_predictions.csv`. In the given example, the inference results are saved as run-length encoding using the custom `SaveRunLengthEncodingd` transform. The format of the resulting CSV file will be:
    ```csv
    Id,Predicted
    liver_xx_xx.nii,0 1 2 3
    liver_xx_xx.nii,0 1 2 3
    ```
    Here, the `Id` column contains the filename of the image being predicted, and the `Predicted` column contains the run-length encoded segmentation mask for that image.

# Manafaln: A simple tool for model training with MONAI & Pytorch-lightning

## Features
 - Configuration based training
 - High flexibility

## Install

Simply download with git and install.
```
git clone https://gitlab.com/nanaha1003/manafaln.git
cd manafaln
python setup.py install
```

Or use our Dockerfile to build a container.
```
git clone https://gitlab.com/nanaha1003/manafaln.git
cd manafaln
docker build -t manafaln:latest .
```

## Usage

### Recommanded Directory Layout

You can find examples in `examples` directory.

```
project_root
├── config.json
├── commands
├── models
└── ...
```

### Built-in Commands

 - train
 - validate
 - inference

### Configuration File Structure

The configuration file is `config.json`, everything related to current training is written inside.
Three major components are in `config.json`, `trainer`, `data` and `workflow`, the sample layout is as:

```
{
    "trainer": {
        ...
    },
    "data": {
        ...
    },
    "workflow": {
        ...
    }
}
```

The details of each component will be introduced in following subsections.

#### Trainer configuration

The `trainer` section contains two parts: `settings` and `callbacks`. The options in settings will be directly use to contryct `pytorch_lightning.Trainer`, you can find all options from PyTorch Lightning documentation. The `callbacks` are the PyTorch Lightning callback objects for the trainer.

```
"trainer": {
  "settings": {
    "accelerator": "ddp",
    "gpus": 4,
    "benchmark": true,
    "amp_backend": "apex",
    "amp_level": "O2",

    "max_steps": 25000,
    "check_val_every_n_epoch": 10,
    "terminate_on_nan": true,

    "auto_lr_find": false,

    "default_root_dir": "models",
    "log_every_n_steps": 10,
    "resume_from_checkpoint": null
  },

  "callbacks": [
    {
      "name": "LearningRateMonitor"
    },
    {
      "name": "ModelCheckpoint",
      "args": {
        "filename": "best_model",
        "monitor": "val_meandice",
        "mode": "max",
        "save_last": true,
        "save_top_k": 1
      }
    }
  ]
}
```

#### Data configurations

The `data` section contains all informations to build a PyTorch Lightning `LightningDataModule`.

```
"data": {
    "name": "DecathlonDataModule",

    "settings": {
        "data_root": "",
        "data_list": "",
        "is_segmentation": true,

        "use_shm_cache": false,
        "shm_cache_path": "/dev/shm"
    },

    "training": { ... },
    "validation": { ... },
    "test": { ... }
}
```

#### Workflow configurations

xxx

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
 - validate (TODO)
 - export (TODO)
 - inference (TODO)

### Configuration File Structure

The configuration file is `config.json`, everything related to current training is written inside.
Three major components are in `config.json`, `trainer`, `data` and `workflow`.

from os import PathLike

from ruamel.yaml import YAML

def load_yaml(yaml_path: PathLike):
    loader = YAML()
    with open(yaml_path, "r") as f:
        yaml_dict = loader.load(f)
    return yaml_dict

def save_yaml(yaml_path: PathLike, yaml_dict: dict):
    saver = YAML()
    with open(yaml_path, 'w') as f:
        saver.dump(yaml_dict, f)

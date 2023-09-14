from os import PathLike

from ruamel.yaml import YAML


def load_yaml(yaml_path: PathLike) -> dict:
    """
    Load a YAML file and return its content as a dictionary.

    Parameters:
    - yaml_path (PathLike): The path to the YAML file.

    Returns:
    - dict: The content of the YAML file as a dictionary.
    """
    loader = YAML()
    with open(yaml_path, "r") as f:
        yaml_dict = loader.load(f)
    return yaml_dict


def save_yaml(yaml_path: PathLike, yaml_dict: dict) -> None:
    """
    Save a dictionary as a YAML file.

    Parameters:
    - yaml_path (PathLike): The path to save the YAML file.
    - yaml_dict (dict): The dictionary to be saved as YAML.

    Returns:
    - None
    """
    saver = YAML()
    with open(yaml_path, "w") as f:
        saver.dump(yaml_dict, f)

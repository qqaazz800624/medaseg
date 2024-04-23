import torch

def FromTorchScript(
    model_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Loads a TorchScript model from the given file path and returns it as a torch.nn.Module.

    Args:
        model_path (str): The file path of the TorchScript model.
        device (str, optional): The device to load the model on. Defaults to "cpu".

    Returns:
        torch.nn.Module: The loaded TorchScript model.

    """
    model = torch.jit.load(model_path, map_location=device)
    return model


def load_mednext_legacy_state_dict(
    model: torch.nn.Module,
    legacy_state_dict: Dict,
    strict: bool = True
) -> torch.nn.Module:
    """
    Convert legacy (officail) MedNeXt state_dict to the manafaln MedNeXt.

    Args:
        model: instance of manafaln MedNeXt model.
        legacy_state_dict: The source state_dict from legacy checkpoints.
        strict: if key mismatch is allowed while loading.

    Reutrns:
        torch.nn.Module: The MedNeXt model that loads with given state_dict.
    """
    depth = model.depth

    state_dict = {}
    for key in legacy_state_dict.keys():
        components = key.split(".")
        new_key = []

        if "dummy_tensor" == key:
            continue

        # stem
        if "stem" in key:
            new_key = components

        # enc_blocks
        if "enc" in key:
            new_key.append("enc_blocks")
            new_key.append(components[0].split("_")[2])
            new_key.append(components[1])
            new_key.append("layers")
            new_key.append(components[2])
            new_key.append(components[3])

        # down_blocks
        if "down" in key:
            new_key.append("down_blocks")
            new_key.append(components[0].split("_")[1])
            if not "res_conv" in key:
                new_key.append("layers")
            new_key.append(components[1])
            new_key.append(components[2])

        # bottleneck
        if "bottleneck" in key:
            new_key.append(components[0])
            new_key.append(components[1])
            new_key.append("layers")
            new_key.append(components[2])
            new_key.append(components[3])

        # up_blocks
        if "up" in key:
            new_key.append("up_blocks")
            new_key.append(str(depth - 1 - int(components[0].split("_")[1])))
            if not "res_conv" in key:
                new_key.append("layers")
            new_key.append(components[1])
            new_key.append(components[2])

        # dec_blocks
        if "dec" in key:
            new_key.append("dec_blocks")
            new_key.append(str(depth - 1 - int(components[0].split("_")[2])))
            new_key.append(components[1])
            new_key.append("layers")
            new_key.append(components[2])
            new_key.append(components[3])

        # out & ds_out_blocks
        if "out" in key:
            if "out_0" in key:
                new_key.extend(["out", "conv", components[2]])
            else:
                new_key.append("ds_out_blocks")
                new_key.append(str(depth - int(components[0].split("_")[1])))
                new_key.append("conv")
                new_key.append(components[2])

        new_key = ".".join(new_key)
        state_dict[new_key] = legacy_state_dict[key]

    model.load_state_dict(state_dict, strict=strict)
    return model


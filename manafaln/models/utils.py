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


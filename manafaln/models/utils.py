import torch

def FromTorchScript(
    model_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    model = torch.jit.load(model_path, map_location=device)
    return model


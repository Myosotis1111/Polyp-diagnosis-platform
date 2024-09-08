import torch
from lib.pvt import PolypPVT


def load_seg_model(pth_path, device='cpu'):
    """
    Load the PolypPVT model with specified weights and device.

    Args:
        pth_path (str): Path to the model weights.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        model: The loaded PolypPVT model.
    """
    # Initialize the model
    model = PolypPVT()

    # Load the model state dictionary with map_location set to the specified device
    model.load_state_dict(torch.load(pth_path, map_location=torch.device(device)))

    # Ensure the model is on the specified device
    model.to(device)
    model.eval()
    print(f"Segmentation model loaded successfully from {pth_path}")

    return model

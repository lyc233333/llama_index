"""Utility functions for MobileCLIP embeddings."""

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


def preprocess_image(image_path: str, device: str) -> torch.Tensor:
    """
    Preprocess an image for MobileCLIP model.
    
    Args:
        image_path: Path to the image file
        device: Device to load the image to ('cpu' or 'cuda')
        
    Returns:
        Preprocessed image tensor
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "MobileCLIP utilities require `pip install open-clip-torch`."
        )
    
    _, _, preprocess = open_clip.create_model_and_transforms(
        model_name="MobileCLIP-S2", device=device
    )
    
    image = Image.open(image_path)
    processed_image = preprocess(image).unsqueeze(0).to(device)
    
    return processed_image

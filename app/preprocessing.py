"""
Image preprocessing utilities for the lung cancer detection model.
"""
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def preprocess_for_model(image: Image.Image) -> np.ndarray:
    """
    Preprocesses a PIL Image for the TorchXRayVision model.
    
    TorchXRayVision models expect:
    - Grayscale images
    - 224x224 resolution
    - Pixel values in [-1024, 1024] range
    """
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Resize to 224x224
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)

        # Scale to [0, 1] first
        img_array = img_array / 255.0
        
        # Scale to [-1024, 1024] as expected by TorchXRayVision
        img_array = (img_array * 2048) - 1024

        # Add channel dimension: (224, 224) -> (1, 224, 224)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        logger.error(f"Failed during image preprocessing: {str(e)}")
        raise
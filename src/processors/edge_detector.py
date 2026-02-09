"""Edge detection - pure functional approach.

Provides various edge detection algorithms as pure functions.
"""
import numpy as np
from scipy import ndimage
from typing import Optional

from ..models import Bitmap


def detect_edges_canny(
    bitmap: Bitmap,
    low_threshold: float = 50,
    high_threshold: float = 150
) -> Bitmap:
    """Apply Canny edge detection.
    
    Args:
        bitmap: Input bitmap (will be converted to grayscale)
        low_threshold: Lower threshold for edge linking
        high_threshold: Upper threshold for edge detection
        
    Returns:
        Binary edge map as Bitmap
    """
    from skimage import feature
    
    # Ensure grayscale
    if not bitmap.is_grayscale:
        from .color_quantizer import to_grayscale
        bitmap = to_grayscale(bitmap)
    
    # Apply Canny
    edges = feature.canny(
        bitmap.data,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    
    # Convert boolean to uint8
    edge_data = (edges * 255).astype(np.uint8)
    
    return Bitmap(
        width=bitmap.width,
        height=bitmap.height,
        data=edge_data,
        mode='L'
    )


def detect_edges_sobel(bitmap: Bitmap) -> Bitmap:
    """Apply Sobel edge detection.
    
    Args:
        bitmap: Input bitmap (will be converted to grayscale)
        
    Returns:
        Edge magnitude bitmap
    """
    # Ensure grayscale
    if not bitmap.is_grayscale:
        from .color_quantizer import to_grayscale
        bitmap = to_grayscale(bitmap)
    
    # Sobel filters
    sx = ndimage.sobel(bitmap.data, axis=0)
    sy = ndimage.sobel(bitmap.data, axis=1)
    
    # Calculate magnitude
    magnitude = np.hypot(sx, sy)
    magnitude = magnitude / magnitude.max() * 255
    magnitude = magnitude.astype(np.uint8)
    
    return Bitmap(
        width=bitmap.width,
        height=bitmap.height,
        data=magnitude,
        mode='L'
    )


def threshold_bitmap(
    bitmap: Bitmap,
    threshold: int = 128,
    inverse: bool = False
) -> Bitmap:
    """Apply threshold to create binary image.
    
    Args:
        bitmap: Input bitmap
        threshold: Threshold value (0-255)
        inverse: If True, invert the result
        
    Returns:
        Binary bitmap
    """
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    
    # Ensure grayscale
    if not bitmap.is_grayscale:
        from .color_quantizer import to_grayscale
        bitmap = to_grayscale(bitmap)
    
    # Apply threshold
    if inverse:
        binary_data = (bitmap.data < threshold).astype(np.uint8) * 255
    else:
        binary_data = (bitmap.data >= threshold).astype(np.uint8) * 255
    
    return Bitmap(
        width=bitmap.width,
        height=bitmap.height,
        data=binary_data,
        mode='L'
    )


def despeckle(bitmap: Bitmap, min_size: int = 4) -> Bitmap:
    """Remove small noise regions from binary image.
    
    Args:
        bitmap: Binary bitmap
        min_size: Minimum region size to keep
        
    Returns:
        Despeckled bitmap
    """
    from skimage import morphology
    
    # Ensure binary
    binary = bitmap.data > 128
    
    # Remove small objects
    cleaned = morphology.remove_small_objects(binary, min_size=min_size)
    
    # Convert back to uint8
    cleaned_data = (cleaned * 255).astype(np.uint8)
    
    return Bitmap(
        width=bitmap.width,
        height=bitmap.height,
        data=cleaned_data,
        mode='L'
    )
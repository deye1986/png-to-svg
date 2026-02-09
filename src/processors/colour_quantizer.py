"""Color quantization - pure functional approach.

Reduces image colors to a smaller palette for easier vectorization.
"""
import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans

from ..models import Bitmap


def quantize_colors(bitmap: Bitmap, n_colors: int) -> Bitmap:
    """Reduce image to n_colors using k-means clustering.
    
    Pure function: returns new Bitmap without modifying input.
    
    Args:
        bitmap: Input bitmap
        n_colors: Target number of colors
        
    Returns:
        New bitmap with quantized colors
    """
    # Reshape to (pixels, channels)
    pixels = bitmap.data.reshape(-1, bitmap.data.shape[-1])
    
    # Handle grayscale
    if bitmap.is_grayscale:
        pixels = pixels.reshape(-1, 1)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Get cluster centers (quantized colors)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    
    # Map each pixel to its cluster center
    quantized_pixels = centers[labels]
    
    # Reshape back to original dimensions
    quantized_data = quantized_pixels.reshape(bitmap.data.shape)
    
    return Bitmap(
        width=bitmap.width,
        height=bitmap.height,
        data=quantized_data,
        mode=bitmap.mode
    )


def posterize(bitmap: Bitmap, levels: int = 4) -> Bitmap:
    """Posterize image by reducing color levels per channel.
    
    Faster alternative to k-means for simple color reduction.
    
    Args:
        bitmap: Input bitmap
        levels: Number of levels per channel (2-256)
        
    Returns:
        New bitmap with posterized colors
    """
    if not 2 <= levels <= 256:
        raise ValueError("levels must be between 2 and 256")
    
    # Calculate step size
    step = 256 // levels
    
    # Posterize each channel
    posterized_data = (bitmap.data // step) * step
    posterized_data = posterized_data.astype(np.uint8)
    
    return Bitmap(
        width=bitmap.width,
        height=bitmap.height,
        data=posterized_data,
        mode=bitmap.mode
    )


def to_grayscale(bitmap: Bitmap) -> Bitmap:
    """Convert bitmap to grayscale.
    
    Args:
        bitmap: Input bitmap
        
    Returns:
        Grayscale bitmap
    """
    if bitmap.is_grayscale:
        return bitmap
    
    # Standard luminance conversion
    if bitmap.mode == 'RGBA':
        gray_data = np.dot(bitmap.data[..., :3], [0.299, 0.587, 0.114])
    else:
        gray_data = np.dot(bitmap.data, [0.299, 0.587, 0.114])
    
    gray_data = gray_data.astype(np.uint8)
    
    return Bitmap(
        width=bitmap.width,
        height=bitmap.height,
        data=gray_data,
        mode='L'
    )
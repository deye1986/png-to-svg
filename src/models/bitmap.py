"""Bitmap data model - immutable representation of image data."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Bitmap:
    """Immutable bitmap representation.
    
    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        data: Numpy array of pixel data (H x W x C)
        mode: Colour mode ('RGB', 'RGBA', 'L', etc.)
    """
    width: int
    height: int
    data: np.ndarray
    mode: str
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Returns (height, width) tuple."""
        return (self.height, self.width)
    
    @property
    def is_grayscale(self) -> bool:
        """Check if bitmap is grayscale."""
        return self.mode == 'L'
    
    def __post_init__(self):
        """Validate bitmap data."""
        if self.data.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Data shape {self.data.shape} doesn't match "
                f"dimensions {self.height}x{self.width}"
            )
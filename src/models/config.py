"""Configuration data model."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for PNG to SVG conversion.
    
    Attributes:
        colour_count: Target number of colours for quantization
        smoothness: Curve smoothing factor (0.0 - 1.0)
        threshold: Edge detection threshold (0-255)
        despeckle: Remove noise pixels (area < this value)
        optimize_paths: Apply path optimization
    """
    color_count: int = 16
    smoothness: float = 0.5
    threshold: int = 128
    despeckle: int = 4
    optimize_paths: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 2 <= self.color_count <= 256:
            raise ValueError("colour_count must be between 2 and 256")
        if not 0.0 <= self.smoothness <= 1.0:
            raise ValueError("smoothness must be between 0.0 and 1.0")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
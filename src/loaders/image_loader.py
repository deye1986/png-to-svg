from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image

from ..models import Bitmap


class ImageLoader:
    """Handles loading PNG images into Bitmap representation.
    
    This class encapsulates PIL-specific operations and provides
    a clean interface for loading images.
    """
    
    @staticmethod
    def load(file_path: Union[str, Path]) -> Bitmap:
        """Load a PNG image from file.
        
        Args:
            file_path: Path to PNG file
            
        Returns:
            Bitmap representation of the image
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid image
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        try:
            with Image.open(file_path) as img:
                # Convert to RGB or RGBA
                if img.mode not in ('RGB', 'RGBA', 'L'):
                    img = img.convert('RGB')
                
                # Convert to numpy array
                data = np.array(img)
                
                return Bitmap(
                    width=img.width,
                    height=img.height,
                    data=data,
                    mode=img.mode
                )
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    @staticmethod
    def from_pil_image(pil_image: Image.Image) -> Bitmap:
        """Create Bitmap from existing PIL Image.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Bitmap representation
        """
        data = np.array(pil_image)
        return Bitmap(
            width=pil_image.width,
            height=pil_image.height,
            data=data,
            mode=pil_image.mode
        )
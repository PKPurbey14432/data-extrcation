
import numpy as np
from PIL import Image
from rembg import remove, new_session
from typing import Optional, Union
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackgroundRemover:
    """
    Removes background from images using deep learning models.
    """
    
    def __init__(self, model_name: str = 'u2net'):
        """
        Initialize background remover.
        
        Args:
            model_name: Model to use ('u2net', 'u2netp', 'u2net_human_seg', etc.)
        """
        self.model_name = model_name
        try:
            self.session = new_session(model_name)
        except Exception as e:
            logger.warning(f"Could not create session with model {model_name}: {e}. Will use default.")
            self.session = None
        logger.info(f"Background remover initialized with model: {model_name}")
    
    def remove_background(self, 
                         input_path: str, 
                         output_path: Optional[str] = None,
                         return_image: bool = False) -> Optional[np.ndarray]:
        """
        Remove background from image.
        
        Args:
            input_path: Path to input image
            output_path: Optional path to save output image
            return_image: Whether to return the processed image as numpy array
            
        Returns:
            Processed image as numpy array if return_image=True, else None
        """
        logger.info(f"Removing background from {input_path}")
        
        try:
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
            
            if self.session is not None:
                output_data = remove(input_data, session=self.session)
            else:
                try:
                    output_data = remove(input_data, model_name=self.model_name)
                except TypeError:
                    output_data = remove(input_data)
            
            output_image = Image.open(io.BytesIO(output_data))
            
            if output_path:
                output_image.save(output_path)
                logger.info(f"Background removed image saved to {output_path}")
            
            if return_image:
                return np.array(output_image)
            
            return None
            
        except Exception as e:
            logger.error(f"Error removing background: {str(e)}")
            raise
    
    def remove_background_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Remove background from image array (in-memory processing).
        
        Args:
            image_array: Input image as numpy array
            
        Returns:
            Image with background removed as numpy array
        """
        pil_image = Image.fromarray(image_array)
        
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        if self.session is not None:
            output_data = remove(img_byte_arr, session=self.session)
        else:
            try:
                output_data = remove(img_byte_arr, model_name=self.model_name)
            except TypeError:
                output_data = remove(img_byte_arr)
        
        output_image = Image.open(io.BytesIO(output_data))
        return np.array(output_image)
    
    def remove_background_with_transparency(self, 
                                           input_path: str, 
                                           output_path: str,
                                           format: str = 'PNG') -> None:
        """
        Remove background and save with transparency (alpha channel).
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            format: Output format (PNG recommended for transparency)
        """
        logger.info(f"Removing background with transparency from {input_path}")
        
        with open(input_path, 'rb') as input_file:
            input_data = input_file.read()
        
        if self.session is not None:
            output_data = remove(input_data, session=self.session)
        else:
            try:
                output_data = remove(input_data, model_name=self.model_name)
            except TypeError:
                output_data = remove(input_data)
        
        output_image = Image.open(io.BytesIO(output_data))
        output_image.save(output_path, format=format)
        
        logger.info(f"Transparent image saved to {output_path}")


if __name__ == "__main__":
    remover = BackgroundRemover()
    remover.remove_background("/home/lnv221/Pictures/Screenshots/Screenshot from 2026-01-12 19-06-20.png", "output_no_bg.png")

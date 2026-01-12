

import cv2
import numpy as np
from PIL import Image
import easyocr
from paddleocr import PaddleOCR
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRTextExtractor:
    """
    Extracts text from images including vertical, horizontal, and embossed text.
    Uses multiple OCR engines for better accuracy.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize OCR extractor with multiple engines.
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
        """
        logger.info("Initializing OCR engines...")
        self.easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
        
        self.paddleocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        logger.info("OCR engines initialized successfully")
    
    def preprocess_image(self, image_path: str, enhance_embossed: bool = True) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy, especially for embossed text.
        
        Args:
            image_path: Path to the image file
            enhance_embossed: Whether to apply embossed text enhancement
            
        Returns:
            Preprocessed image as numpy array
        """
        import os
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found: {image_path}\n"
                f"Please check the file path. Common issues:\n"
                f"  - File doesn't exist at this location\n"
                f"  - Incorrect file extension (e.g., .png.jpg instead of .png)\n"
                f"  - File path has typos"
            )
        
        if not os.path.isfile(image_path):
            raise ValueError(f"Path exists but is not a file: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(
                f"Could not read image from {image_path}\n"
                f"This usually means:\n"
                f"  - File is corrupted\n"
                f"  - Unsupported image format\n"
                f"  - File permissions issue\n"
                f"Supported formats: JPEG, PNG, BMP, TIFF, etc."
            )
        
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if enhance_embossed:
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            kernel = np.ones((2, 2), np.uint8)
            enhanced = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            
            combined = cv2.addWeighted(gray, 0.7, enhanced, 0.3, 0)
            
            kernel_sharpen = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
            sharpened = cv2.filter2D(combined, -1, kernel_sharpen)
            
            return sharpened
        
        return gray
    
    def extract_with_easyocr(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text using EasyOCR (good for multiple orientations).
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            List of dictionaries with text, confidence, and bounding box info
        """
        results = self.easyocr_reader.readtext(image)
        
        extracted_texts = []
        for (bbox, text, confidence) in results:
            extracted_texts.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox,
                'engine': 'easyocr'
            })
        
        return extracted_texts
    
    def extract_with_paddleocr(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text using PaddleOCR (excellent for vertical text).
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            List of dictionaries with text, confidence, and bounding box info
        """

        try:
            if hasattr(self.paddleocr, 'predict'):
                results = self.paddleocr.predict(image)
            else:
                results = self.paddleocr.ocr(image)
        except Exception as e:
            logger.warning(f"PaddleOCR extraction failed: {str(e)}")
            return []
        
        extracted_texts = []
        if results:
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list) and len(results[0]) > 0:
                    if isinstance(results[0][0], list):
                        results = results[0]
                
                for line in results:
                    if line:
                        if isinstance(line, list) and len(line) >= 2:
                            bbox = line[0]
                            text_info = line[1]
                            
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text, confidence = text_info[0], text_info[1]
                            elif isinstance(text_info, str):
                                text, confidence = text_info, 1.0
                            else:
                                text, confidence = str(text_info), 1.0
                            
                            extracted_texts.append({
                                'text': text,
                                'confidence': float(confidence) if confidence else 1.0,
                                'bbox': bbox,
                                'engine': 'paddleocr'
                            })
        
        return extracted_texts
    
    def detect_text_orientation(self, bbox: List) -> str:
        """
        Detect if text is vertical or horizontal based on bounding box.
        
        Args:
            bbox: Bounding box coordinates
            
        Returns:
            'vertical' or 'horizontal'
        """
        if len(bbox) < 2:
            return 'horizontal'
        
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        if height > width * 1.5:
            return 'vertical'
        return 'horizontal'
    
    def extract_text(self, image_path: str) -> Dict:
        """
        Main method to extract all text from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all extracted text with metadata
        """
        logger.info(f"Extracting text from {image_path}")
        
        preprocessed = self.preprocess_image(image_path, enhance_embossed=True)
        
        easyocr_results = self.extract_with_easyocr(preprocessed)
        paddleocr_results = self.extract_with_paddleocr(preprocessed)
        
        all_results = []
        seen_texts = set()
        
        for result in easyocr_results + paddleocr_results:
            text_lower = result['text'].lower().strip()
            if text_lower and text_lower not in seen_texts:
                result['orientation'] = self.detect_text_orientation(result['bbox'])
                all_results.append(result)
                seen_texts.add(text_lower)
        
        all_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        all_text = ' '.join([r['text'] for r in all_results])
        
        horizontal_texts = [r for r in all_results if r['orientation'] == 'horizontal']
        vertical_texts = [r for r in all_results if r['orientation'] == 'vertical']
        
        return {
            'all_text': all_text,
            'all_results': all_results,
            'horizontal_texts': [r['text'] for r in horizontal_texts],
            'vertical_texts': [r['text'] for r in vertical_texts],
            'embossed_texts': all_results, 
            'total_detections': len(all_results),
            'average_confidence': np.mean([r['confidence'] for r in all_results]) if all_results else 0
        }


if __name__ == "__main__":
    extractor = OCRTextExtractor()
    image_path = "/home/lnv221/Pictures/Screenshots/Screenshot from 2026-01-12 19-06-20.png"
    try:
        result = extractor.extract_text(image_path)
        print(f"Extracted Text: {result['all_text']}")
        print(f"Horizontal: {result['horizontal_texts']}")
        print(f"Vertical: {result['vertical_texts']}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

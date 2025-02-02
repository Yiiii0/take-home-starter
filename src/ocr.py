import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings from environment
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "40.0"))
USE_EASYOCR_FALLBACK = os.getenv("USE_EASYOCR_FALLBACK", "false").lower() == "true"
MAX_PAGES = int(os.getenv("MAX_PAGES", "10"))  # Maximum pages to process for testing

def remove_borders(image: np.ndarray) -> np.ndarray:
    """Remove black borders from scanned image."""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Threshold the image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (main content)
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Add padding
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(image.shape[1] - x, w + 2*pad)
            h = min(image.shape[0] - y, h + 2*pad)
            
            # Crop the image
            return image[y:y+h, x:x+w]
    except Exception as e:
        logger.warning(f"Border removal failed: {str(e)}")
    
    return image

def deskew(image: np.ndarray) -> np.ndarray:
    """Deskew the image using Hough Line Transform."""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Apply edge detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        if lines is not None:
            # Calculate the average angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:  # Only rotate if skew is significant
                    # Rotate the image
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(
                        image, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    return rotated

    except Exception as e:
        logger.warning(f"Deskewing failed: {str(e)}")
    
    return image

def enhance_resolution(image: np.ndarray) -> np.ndarray:
    """Enhance image resolution using super resolution."""
    try:
        # Convert to PIL Image for enhancement
        pil_image = Image.fromarray(image)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        sharpened = enhancer.enhance(2.0)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(sharpened)
        contrasted = enhancer.enhance(1.5)
        
        # Convert back to numpy array
        return np.array(contrasted)
    except Exception as e:
        logger.warning(f"Resolution enhancement failed: {str(e)}")
        return image

def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply advanced denoising."""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
        
        return denoised
    except Exception as e:
        logger.warning(f"Denoising failed: {str(e)}")
        return image

def adaptive_binarization(image: np.ndarray) -> np.ndarray:
    """Apply adaptive binarization with optimized parameters."""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding with optimized parameters
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,  # Larger block size for better adaptation
            15   # Higher constant for better contrast
        )
        
        # Apply morphological operations to clean up the result
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    except Exception as e:
        logger.warning(f"Binarization failed: {str(e)}")
        return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Enhanced image preprocessing pipeline for OCR, especially for image PDFs.
    """
    try:
        # Remove borders first
        image = remove_borders(image)
        
        # Enhance resolution
        image = enhance_resolution(image)
        
        # Deskew the image
        image = deskew(image)
        
        # Denoise
        image = denoise_image(image)
        
        # Apply adaptive binarization
        image = adaptive_binarization(image)
        
        return image

    except Exception as e:
        logger.warning(f"Image preprocessing failed: {str(e)}")
        return image

def get_tesseract_config() -> str:
    """Get optimized Tesseract configuration."""
    config = []
    
    # Use Legacy + LSTM OCR Engine Mode (more accurate but slower)
    config.append("--oem 1")
    
    # Page segmentation mode: Automatic page segmentation with orientation detection
    config.append("--psm 3")
    
    # Additional parameters for better accuracy
    config.extend([
        # Improve quality settings
        "-c", "tessedit_do_invert=0",
        "-c", "textord_heavy_nr=1",
        "-c", "textord_min_linesize=3",
        
        # Character whitelist (allow only these characters)
        "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()/-'\"",
        
        # Language settings
        "-l", "eng",  # English language
        
        # DPI setting
        "--dpi", "300"
    ])
    
    return " ".join(config)

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from PDF with enhanced OCR processing.
    For testing, only processes the first MAX_PAGES pages.
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Processing PDF: {pdf_path}")
        pages = []
        
        # Convert PDF to images with higher DPI for better quality
        images = convert_from_path(
            pdf_path,
            first_page=1,
            last_page=MAX_PAGES,
            dpi=300,  # Higher DPI for better quality
            grayscale=True,  # Convert to grayscale during PDF conversion
            thread_count=4,  # Use multiple threads for conversion
            size=(2000, None)  # Resize to larger width for better quality
        )
        
        logger.info(f"Processing first {len(images)} pages for testing")
        
        # Get optimized Tesseract configuration
        try:
            custom_config = get_tesseract_config()
            logger.info(f"Using Tesseract config: {custom_config}")
        except Exception as e:
            logger.warning(f"Failed to get custom config, using default: {str(e)}")
            custom_config = r'--oem 1 --psm 3'
        
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}/{len(images)}")
            
            try:
                # Convert PIL Image to numpy array
                image_np = np.array(image)
                
                # Enhanced preprocessing
                processed_image = preprocess_image(image_np)
                
                # Save intermediate image for debugging if needed
                debug_path = Path(f"test_output/debug_page_{page_num}.png")
                if debug_path.parent.exists():
                    cv2.imwrite(str(debug_path), processed_image)
                
                # Extract text with Tesseract
                try:
                    data = pytesseract.image_to_data(
                        processed_image,
                        output_type=pytesseract.Output.DICT,
                        config=custom_config
                    )
                    
                    # Calculate confidence
                    confidences = [float(conf) for conf in data['conf'] if conf != '-1']
                    confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Get text with improved settings
                    text = pytesseract.image_to_string(
                        processed_image,
                        config=custom_config
                    )
                    
                except Exception as ocr_error:
                    logger.error(f"OCR failed with custom config: {str(ocr_error)}")
                    logger.info("Falling back to default configuration")
                    
                    # Fallback to simpler configuration
                    text = pytesseract.image_to_string(processed_image)
                    confidence = 0.0
                
                # Store results
                page_info = {
                    "page_number": page_num,
                    "text": text.strip(),
                    "confidence": confidence,
                    "source": "tesseract"
                }
                pages.append(page_info)
                
                logger.info(f"Completed page {page_num} with confidence: {confidence:.2f}")
                
            except Exception as page_error:
                logger.error(f"Failed to process page {page_num}: {str(page_error)}")
                # Add empty result for failed page
                pages.append({
                    "page_number": page_num,
                    "text": "",
                    "confidence": 0.0,
                    "source": "tesseract",
                    "error": str(page_error)
                })
            
        return pages
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        try:
            results = extract_text_from_pdf(pdf_path)
            for page in results:
                print(f"\nPage {page['page_number']} "
                      f"(Confidence: {page['confidence']:.2f}%):")
                print("-" * 80)
                print(page['text'])
                print("-" * 80)
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            sys.exit(1)
    else:
        print("Please provide a PDF file path as argument")
        sys.exit(1)

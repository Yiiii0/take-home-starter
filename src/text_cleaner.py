import logging
import re
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common medical-legal abbreviations and their expansions
MEDICAL_ABBREVIATIONS = {
    'dx': 'diagnosis',
    'hx': 'history',
    'tx': 'treatment',
    'sx': 'symptoms',
    'px': 'physical examination',
    'rx': 'prescription',
    'fx': 'fracture',
    'MMI': 'Maximum Medical Improvement',
    'IME': 'Independent Medical Examination',
    'QME': 'Qualified Medical Evaluator',
    'AME': 'Agreed Medical Evaluator',
    'TTD': 'Temporary Total Disability',
    'TPD': 'Temporary Partial Disability',
    'P&S': 'Permanent and Stationary',
    'DOI': 'Date of Injury',
    'WC': "Worker's Compensation",
    'ADL': 'Activities of Daily Living',
    'ROM': 'Range of Motion',
}

def expand_abbreviations(text: str, abbreviations: Dict[str, str] = None) -> str:
    """Expand medical and legal abbreviations in the text."""
    if not text.strip():
        return text
        
    if abbreviations is None:
        abbreviations = MEDICAL_ABBREVIATIONS
        
    # Sort by length (longest first) to avoid partial matches
    sorted_abbrevs = sorted(abbreviations.items(), key=lambda x: len(x[0]), reverse=True)
    
    for abbrev, expansion in sorted_abbrevs:
        # Only replace if it's a whole word
        pattern = fr'\b{re.escape(abbrev)}\b'
        text = re.sub(pattern, f"{abbrev} ({expansion})", text, flags=re.IGNORECASE)
        
    return text

def fix_line_breaks(text: str) -> str:
    """Fix common line break issues in OCR output."""
    if not text.strip():
        return text
        
    # Join hyphenated words split across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Preserve paragraph breaks but remove unnecessary line breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Standardize multiple newlines to double newlines for paragraphs
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def clean_whitespace(text: str) -> str:
    """Clean up whitespace issues."""
    if not text.strip():
        return text
        
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    text = re.sub(r'([({])\s+', r'\1', text)
    
    # Ensure single space after punctuation
    text = re.sub(r'([.,;:!?])\s*', r'\1 ', text)
    
    return text.strip()

def fix_common_ocr_errors(text: str) -> str:
    """Fix common OCR errors and character confusions."""
    if not text.strip():
        return text
        
    try:
        replacements = {
            # Common OCR mistakes
            r'[|]': 'I',  # Vertical bar to I
            r'0(?=\D)': 'O',  # Zero to O when not followed by digit
            r'(?<=\D)0': 'O',  # Zero to O when not preceded by digit
            r'1(?=\D)': 'I',  # One to I when not followed by digit
            r'\bI\b(?=\d)': '1',  # I to 1 when followed by numbers
            r'(?<=\d)l': '1',  # lowercase L to 1 when preceded by numbers
            
            # Fix common medical/legal terms
            r'\b[Pp]atient\b': 'patient',
            r'\b[Dd]octor\b': 'doctor',
            r'\b[Dd]iagnosis\b': 'diagnosis',
            r'\b[Tt]reatment\b': 'treatment',
            
            # Fix common symbols
            r'°': ' degrees',
            r'±': ' plus/minus ',
            r'≤': ' less than or equal to ',
            r'≥': ' greater than or equal to ',
            r'&': 'and',  # Handle ampersands
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    except Exception as e:
        logger.warning(f"Error fixing OCR errors: {str(e)}")
        return text

def normalize_measurements(text: str) -> str:
    """Standardize measurement formats."""
    if not text.strip():
        return text
        
    try:
        # Standardize degree measurements
        text = re.sub(r'(\d+)(?:\s*°|\s+deg\.?|\s+degrees?)', r'\1 degrees', text)
        
        # Standardize ranges
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 to \2', text)
        
        # Standardize percentages
        text = re.sub(r'(\d+)\s*%', r'\1 percent', text)
        
        return text
    except Exception as e:
        logger.warning(f"Error normalizing measurements: {str(e)}")
        return text

def extract_structured_info(text: str) -> Dict[str, str]:
    """Extract structured information from text."""
    if not text.strip():
        return {}
        
    try:
        info = {}
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text)
        if dates:
            info['dates'] = dates
        
        # Extract measurements
        measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:degrees?|mm|cm|inches?|feet|meters?)\b', text)
        if measurements:
            info['measurements'] = measurements
        
        # Extract medical terms (simplified)
        medical_terms = re.findall(r'\b(?:diagnosis|treatment|symptoms?|injury|condition)\b', text.lower())
        if medical_terms:
            info['medical_terms'] = list(set(medical_terms))
        
        return info
    except Exception as e:
        logger.warning(f"Error extracting structured info: {str(e)}")
        return {}

def clean_text(text: str, extract_info: bool = False) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Clean OCR output text with medical-legal domain awareness.
    
    Args:
        text: Input text from OCR
        extract_info: Whether to extract structured information
        
    Returns:
        Tuple of (cleaned text, extracted info dictionary if extract_info=True)
    """
    try:
        logger.info("Starting text cleaning process")
        
        if not text or not text.strip():
            logger.warning("Received empty or whitespace-only text")
            return "", {} if extract_info else None
        
        # Store original length for comparison
        original_length = len(text)
        
        # Basic cleaning
        text = fix_common_ocr_errors(text)
        text = fix_line_breaks(text)
        text = clean_whitespace(text)
        
        # Domain-specific cleaning
        text = expand_abbreviations(text)
        text = normalize_measurements(text)
        
        # Extract structured information if requested
        info = extract_structured_info(text) if extract_info else None
        
        # Log cleaning results
        cleaned_length = len(text)
        logger.info(f"Cleaning complete. Text length: {original_length} -> {cleaned_length}")
        if info:
            logger.info(f"Extracted {len(info)} pieces of structured information")
        
        return text, info
        
    except Exception as e:
        logger.error(f"Error during text cleaning: {str(e)}")
        # Return original text if cleaning fails
        return text.strip() if text else "", None

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Read input from file
        try:
            with open(sys.argv[1], 'r', encoding='utf-8') as f:
                input_text = f.read()
            
            cleaned_text, extracted_info = clean_text(input_text, extract_info=True)
            
            # Write cleaned text to output file
            output_file = sys.argv[1].rsplit('.', 1)[0] + '_cleaned.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            print(f"\nCleaned text written to: {output_file}")
            
            if extracted_info:
                print("\nExtracted Information:")
                for key, value in extracted_info.items():
                    print(f"{key.capitalize()}:")
                    if isinstance(value, list):
                        for item in value:
                            print(f"  - {item}")
                    else:
                        print(f"  {value}")
                        
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Please provide an input text file path as argument")
        sys.exit(1)

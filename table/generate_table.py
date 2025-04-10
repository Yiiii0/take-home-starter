#!/usr/bin/env python3
"""
Medical Document Information Extraction

This script extracts structured information from medical documents (specifically the qme_demo.pdf)
and generates a table with dates, doctor name, descriptions, and source page information.

The approach:
1. Extract text from the PDF directly using PyPDF2 (no OCR needed as the text is selectable)
2. Process each page to identify important medical content, particularly focusing on reviews
3. Use OpenAI API to help extract and categorize the most relevant information
4. Build a structured dataframe with the extracted information
5. Output the results as a CSV and display as a formatted table

The script prioritizes accuracy over speed by using multiple extraction methods and validation.
"""

import sys
import os
import re
from pathlib import Path
import pandas as pd
import datetime
import logging
from typing import Dict, List, Tuple, Optional, Set
import json
from dotenv import load_dotenv

# Try to import PyPDF2
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

# Try to import OpenAI
try:
    import openai
    openai_available = True
except ImportError:
    print("OpenAI not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai
    openai_available = True

# Try to import dotenv for environment variables
try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# Excel support (optional)
try:
    import openpyxl
    excel_available = True
except ImportError:
    excel_available = False
    print("openpyxl not found. Will not create Excel output (only CSV).")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables for OpenAI API
load_dotenv(Path(__file__).parent.parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key loaded successfully")
else:
    logger.warning("OpenAI API key not found in .env file. LLM assistance will be disabled.")
    openai_available = False

# Common patterns for medical document information
DATE_PATTERNS = [
    r'\b(0?[1-9]|1[0-2])[\/\-\.](0?[1-9]|[12][0-9]|3[01])[\/\-\.](19|20)\d{2}\b',  # MM/DD/YYYY
    r'\b(19|20)\d{2}[\/\-\.](0?[1-9]|1[0-2])[\/\-\.](0?[1-9]|[12][0-9]|3[01])\b',  # YYYY/MM/DD
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:[0-9]{1,2}),? (?:19|20)[0-9]{2}\b',  # Month DD, YYYY
]

# Doctor title patterns - enhanced with more variations
DOCTOR_PATTERNS = [
    r'Dr\.\s[A-Z][a-z]+\s[A-Z][a-z]+',  # Dr. John Smith
    r'[A-Z][a-z]+\s[A-Z]\.\s[A-Z][a-z]+,\sM\.D\.',  # John C. Austin, M.D.
    r'[A-Z][a-z]+\s[A-Z][a-z]+,\sMD',  # John Smith, MD
    r'[A-Z][a-z]+\s[A-Z][a-z]+,\sM\.D\.',  # John Smith, M.D.
    r'[A-Z][a-z]+\s[A-Z][a-z]+,\sPh\.D\.',  # John Smith, Ph.D.
    r'[A-Z][a-z]+\s[A-Z][a-z]+\s(?:MD|M\.D\.|Ph\.D\.)',  # John Smith MD
    r'(?:Dr|Doctor)\s[A-Z][a-z]+\s[A-Z][a-z]+',  # Doctor John Smith
    r'(?:orthopedic|medical|shoulder|spine)\s+surgeon\s+(?:Dr\.|Doctor)?\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # medical surgeon Dr. John Smith
]

# Common medical terms to help identify relevant sections
MEDICAL_TERMS = [
    'diagnosis', 'treatment', 'evaluation', 'assessment', 'examination', 
    'injury', 'condition', 'medication', 'therapy', 'surgery', 'procedure',
    'recommendation', 'prescription', 'referral', 'consultation', 'follow-up',
    'MRI', 'X-ray', 'CT scan', 'test', 'result', 'pain', 'symptom',
    'orthopedic', 'surgeon', 'physician', 'doctor', 'medical', 'clinical',
    'patient', 'hospital', 'clinic', 'specialist', 'consultation', 
    'review', 'imaging', 'studies', 'findings', 'history', 'shoulder', 'brain', 
    'spine', 'right', 'left', 'atrophy', 'tear', 'tendon', 'muscle'
]

# Sections likely to contain important information
IMPORTANT_SECTIONS = [
    'Relevant Imaging Studies', 'Medical Record Review', 'Findings', 
    'History', 'Assessment', 'Diagnosis', 'Recommendations', 'Treatment Plan'
]

class MedicalDocumentParser:
    """Parser for extracting structured information from medical documents."""
    
    def __init__(self, pdf_path: str, focus_page: int = 16):
        """
        Initialize the parser with the PDF file path.
        
        Args:
            pdf_path: Path to the PDF file to parse
            focus_page: Page number to start focusing on important content (1-indexed)
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
        self.pages_text = []
        self.extracted_data = []
        self.focus_page = focus_page
    
    def extract_text_from_pdf(self) -> List[str]:
        """
        Extract text from each page of the PDF.
        
        Returns:
            List of strings, one per page of the PDF.
        """
        try:
            logger.info(f"Extracting text from {self.pdf_path}")
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                logger.info(f"PDF has {num_pages} pages")
                
                self.pages_text = []
                for i in range(num_pages):
                    try:
                        page = reader.pages[i]
                        text = page.extract_text()
                        self.pages_text.append(text if text else "")
                        if i % 10 == 0:  # Log progress every 10 pages
                            logger.info(f"Processed {i+1}/{num_pages} pages")
                    except Exception as e:
                        logger.error(f"Error extracting text from page {i+1}: {e}")
                        self.pages_text.append("")
                
                logger.info(f"Successfully extracted text from {num_pages} pages")
                return self.pages_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def remove_header(self, text: str) -> str:
        """
        Remove the standard header from the page text.
        
        Args:
            text: The text of a single page
            
        Returns:
            Text with the header removed
        """
        # Look for patterns like "NAME: THOMPSON, MARK" and "DOB: 05/19/1970" to identify header
        lines = text.split('\n')
        clean_lines = []
        in_header = True
        
        for line in lines:
            # Check if this line is likely part of the header
            if in_header and (
                re.search(r'NAME:|DOB:|DOI:', line) or 
                re.search(r'John C\. Austin, M\.D\.', line) or
                line.strip() == ''
            ):
                continue
            else:
                in_header = False
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def find_dates(self, text: str, ignore_header_dates: bool = True) -> List[str]:
        """
        Find all dates in the given text.
        
        Args:
            text: Text to search for dates
            ignore_header_dates: Whether to ignore dates that appear in the header
            
        Returns:
            List of date strings found in the text
        """
        # If we want to ignore header dates, remove the header first
        if ignore_header_dates:
            # Only remove header-like content from the first few lines
            lines = text.split('\n', 5)
            header_text = '\n'.join(lines[:3])
            body_text = '\n'.join(lines[3:]) if len(lines) > 3 else ""
            
            # Find dates in the body text
            all_dates = []
            for pattern in DATE_PATTERNS:
                matches = re.findall(pattern, body_text)
                if matches:
                    if isinstance(matches[0], tuple):
                        # Some regex patterns return tuples of capture groups
                        for match in matches:
                            # Reconstruct the full date string
                            if len(match) == 3:  # MM/DD/YYYY or similar
                                date_str = f"{match[0]}/{match[1]}/{match[2]}"
                                all_dates.append(date_str)
                    else:
                        # Pattern returns full date strings
                        all_dates.extend(matches)
        else:
            # Find all dates in the text
            all_dates = []
            for pattern in DATE_PATTERNS:
                matches = re.findall(pattern, text)
                if matches:
                    if isinstance(matches[0], tuple):
                        # Some regex patterns return tuples of capture groups
                        for match in matches:
                            # Reconstruct the full date string
                            if len(match) == 3:  # MM/DD/YYYY or similar
                                date_str = f"{match[0]}/{match[1]}/{match[2]}"
                                all_dates.append(date_str)
                    else:
                        # Pattern returns full date strings
                        all_dates.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        return [date for date in all_dates if not (date in seen or seen.add(date))]
    
    def find_doctors(self, text: str) -> List[str]:
        """
        Find all doctor names in the given text.
        
        Args:
            text: Text to search for doctor names
            
        Returns:
            List of doctor names found in the text
        """
        doctors = []
        
        # Use regex patterns to find doctor names
        for pattern in DOCTOR_PATTERNS:
            matches = re.findall(pattern, text)
            doctors.extend(matches)
        
        # Specific search for known doctors from sample
        if "John C. Austin" in text:
            doctors.append("John C. Austin, M.D.")
        
        # Look for specific formats like "evaluated by Dr. X" or "ORDERING PROVIDER: X"
        eval_matches = re.findall(r'(?:evaluated|examined|seen|referred|treated)\s+by\s+(?:Dr\.|Doctor)?\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
        if eval_matches:
            for match in eval_matches:
                doctors.append(f"Dr. {match}")
        
        # Look for ordering providers
        provider_matches = re.findall(r'(?:ORDERING PROVIDER|PROVIDER):\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
        if provider_matches:
            for match in provider_matches:
                if "MD" in text[text.find(match):text.find(match) + len(match) + 10]:
                    doctors.append(f"{match}, MD")
                else:
                    doctors.append(match)
        
        # Remove duplicates while preserving order
        seen = set()
        return [doc for doc in doctors if not (doc in seen or seen.add(doc))]
    
    def identify_section(self, text: str) -> str:
        """
        Identify which medical section this text belongs to.
        
        Args:
            text: Text to analyze
            
        Returns:
            Identified section name or "Unknown"
        """
        for section in IMPORTANT_SECTIONS:
            if section.lower() in text.lower():
                return section
        
        return "Unknown"
    
    def extract_description(self, text: str, page_num: int, max_length: int = 300) -> str:
        """
        Extract a relevant description from text.
        
        Args:
            text: The text to extract a description from
            page_num: The page number (0-indexed)
            max_length: Maximum length of the extracted description
            
        Returns:
            A relevant description from the text
        """
        # Remove the header
        text = self.remove_header(text)
        
        # If this is a page we should focus on (page 16+), give it priority
        is_focus_page = page_num + 1 >= self.focus_page
        
        # Look for paragraphs containing medical terms
        paragraphs = re.split(r'\n\s*\n', text)
        relevant_paragraphs = []
        
        for para in paragraphs:
            if any(term.lower() in para.lower() for term in MEDICAL_TERMS):
                relevant_paragraphs.append(para)
            
            # If this paragraph contains an important section header, give it priority
            if any(section.lower() in para.lower() for section in IMPORTANT_SECTIONS):
                relevant_paragraphs.insert(0, para)
        
        if relevant_paragraphs:
            # Join the relevant paragraphs and truncate if needed
            description = " ".join(relevant_paragraphs)
            if len(description) > max_length:
                description = description[:max_length] + "..."
            return description
        
        # If no relevant paragraphs found, return the beginning of the text
        return text[:max_length] + "..." if len(text) > max_length else text
    
    def use_llm_to_extract_info(self, text: str, page_num: int) -> Dict:
        """
        Use OpenAI's LLM to extract relevant information from the text.
        
        Args:
            text: Text to analyze
            page_num: Page number for context
            
        Returns:
            Dictionary with extracted information
        """
        if not openai_available or not OPENAI_API_KEY:
            logger.warning("OpenAI API not available. Skipping LLM extraction.")
            return {
                "date": "Unknown",
                "doctor_name": "Unknown",
                "description": text[:300] + "..." if len(text) > 300 else text,
                "section": "Unknown"
            }
        
        try:
            # Clean the header from text
            clean_text = self.remove_header(text)
            
            # Prepare a prompt for the API
            prompt = f"""
            Extract the following information from this medical document text (from page {page_num+1}):
            1. Any specific dates mentioned in the content (not in headers)
            2. Doctor names or medical providers mentioned
            3. A concise summary of the key medical information (max 150 words)
            4. The medical section this appears to be from (e.g., "Medical Record Review", "Imaging Studies", "Findings", etc.)

            Text:
            {clean_text[:1500]}  # Limit text to avoid exceeding token limits
            
            Format your response as a JSON object with keys: "date", "doctor_name", "description", "section"
            If a field is not found, use "Unknown" as the value.
            The "date" field should be a single string, not a list. If multiple dates are found, just use the most relevant one.
            """
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical document analyzer that extracts structured information from medical reports. Respond only with the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Process the response
            content = response.choices[0].message.content.strip()
            
            # Extract the JSON object
            json_str = content
            # If content contains markdown code blocks, extract the JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            
            # Parse the JSON
            try:
                result = json.loads(json_str)
                
                # Ensure all required fields exist
                required_fields = ["date", "doctor_name", "description", "section"]
                for field in required_fields:
                    if field not in result:
                        result[field] = "Unknown"
                
                # Ensure all values are strings (not lists)
                for key, value in result.items():
                    if isinstance(value, list):
                        if len(value) > 0:
                            result[key] = ", ".join(str(item) for item in value)
                        else:
                            result[key] = "Unknown"
                
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM output: {content}")
                # Fall back to extracting via regex
                return {
                    "date": "Unknown",
                    "doctor_name": "Unknown", 
                    "description": clean_text[:300] + "..." if len(clean_text) > 300 else clean_text,
                    "section": "Unknown"
                }
                
        except Exception as e:
            logger.error(f"Error using LLM for extraction: {e}")
            return {
                "date": "Unknown",
                "doctor_name": "Unknown",
                "description": text[:300] + "..." if len(text) > 300 else text,
                "section": "Unknown"
            }
    
    def process_page(self, page_num: int) -> List[Dict]:
        """
        Process a single page to extract structured information.
        
        Args:
            page_num: The page number to process (0-indexed)
            
        Returns:
            List of dictionaries, each containing date, doctor, description, and source
        """
        if page_num >= len(self.pages_text):
            logger.warning(f"Page {page_num+1} does not exist in the document")
            return []
        
        text = self.pages_text[page_num]
        if not text.strip():
            logger.debug(f"Page {page_num+1} has no text")
            return []
        
        # Decide if we should use LLM based on whether this is a focus page
        is_focus_page = page_num + 1 >= self.focus_page
        use_llm = is_focus_page and openai_available and OPENAI_API_KEY
        
        entries = []
        
        if use_llm:
            # Use LLM to extract information
            logger.info(f"Using LLM to process page {page_num+1}")
            llm_result = self.use_llm_to_extract_info(text, page_num)
            
            # Ensure all values are strings (not lists or other non-hashable types)
            for key in llm_result:
                if not isinstance(llm_result[key], str):
                    llm_result[key] = str(llm_result[key])
            
            entry = {
                "date": llm_result.get("date", "Unknown"),
                "doctor_name": llm_result.get("doctor_name", "Unknown"),
                "description": llm_result.get("description", "Unknown"),
                "section": llm_result.get("section", "Unknown"),
                "source": f"Page {page_num+1}"
            }
            entries.append(entry)
        else:
            # Extract dates, doctors, and a description using regex patterns
            dates = self.find_dates(text, ignore_header_dates=True)
            doctors = self.find_doctors(text)
            description = self.extract_description(text, page_num)
            section = self.identify_section(text)
            
            # If we have both dates and doctors, create an entry for each combination
            if dates and doctors:
                for date in dates:
                    for doctor in doctors:
                        entries.append({
                            "date": date,
                            "doctor_name": doctor,
                            "description": description,
                            "section": section,
                            "source": f"Page {page_num+1}"
                        })
            # If we have only dates, create an entry for each date
            elif dates:
                for date in dates:
                    entries.append({
                        "date": date,
                        "doctor_name": "Unknown",
                        "description": description,
                        "section": section,
                        "source": f"Page {page_num+1}"
                    })
            # If we have only doctors, create an entry for each doctor
            elif doctors:
                for doctor in doctors:
                    entries.append({
                        "date": "Unknown",
                        "doctor_name": doctor,
                        "description": description,
                        "section": section,
                        "source": f"Page {page_num+1}"
                    })
            # If we have neither, create a single entry
            else:
                entries.append({
                    "date": "Unknown",
                    "doctor_name": "Unknown",
                    "description": description,
                    "section": section,
                    "source": f"Page {page_num+1}"
                })
        
        return entries
    
    def process_document(self) -> pd.DataFrame:
        """
        Process the entire document and build a dataframe of structured information.
        
        Returns:
            DataFrame containing date, doctor_name, description, section, and source columns
        """
        # Extract text if not already done
        if not self.pages_text:
            self.extract_text_from_pdf()
        
        # Process each page
        all_entries = []
        for i in range(len(self.pages_text)):
            entries = self.process_page(i)
            all_entries.extend(entries)
        
        # Create dataframe from extracted data
        df = pd.DataFrame(all_entries)
        
        # Clean and filter the dataframe
        self._clean_dataframe(df)
        
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> None:
        """
        Clean the dataframe to remove duplicates and standardize formats.
        
        Args:
            df: DataFrame to clean (modified in-place)
        """
        if df.empty:
            return
        
        # Ensure all data is hashable by converting any potential lists to strings
        for column in df.columns:
            df[column] = df[column].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        
        # Standardize date format if possible
        if 'date' in df.columns:
            def standardize_date(date_str):
                if date_str == "Unknown":
                    return date_str
                try:
                    # Try different date formats
                    for fmt in ['%m/%d/%Y', '%Y/%m/%d', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y']:
                        try:
                            dt = datetime.datetime.strptime(date_str, fmt)
                            return dt.strftime('%m/%d/%Y')
                        except ValueError:
                            continue
                    return date_str  # Return original if no format matches
                except Exception:
                    return date_str
            
            df['date'] = df['date'].apply(standardize_date)
        
        # Remove exact duplicates
        df.drop_duplicates(inplace=True)
        
        # Remove entries with exactly the same description but different page sources
        df.drop_duplicates(subset=['date', 'doctor_name', 'description'], keep='first', inplace=True)
        
        # Sort the dataframe by source so pages appear in order
        if 'source' in df.columns:
            df['page_num'] = df['source'].str.extract(r'Page (\d+)').astype(int)
            df.sort_values(by=['page_num'], inplace=True)
            df.drop(columns=['page_num'], inplace=True)

def main():
    """Main function to extract table from the qme_demo.pdf file."""
    # Path to the PDF file
    pdf_dir = Path(__file__).parent.parent / "data"
    pdf_path = pdf_dir / "qme_demo.pdf"
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "table_output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Use the MedicalDocumentParser class with focus on page 16 onwards
        parser = MedicalDocumentParser(str(pdf_path), focus_page=16)
        df = parser.process_document()
        
        # Ensure no problematic data types in the DataFrame
        for column in df.columns:
            df[column] = df[column].astype(str)
        
        # Save results to CSV
        csv_path = output_dir / "medical_table.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Table saved to {csv_path}")
        
        # Also save as Excel if possible
        if excel_available:
            try:
                excel_path = output_dir / "medical_table.xlsx"
                df.to_excel(excel_path, index=False)
                logger.info(f"Table also saved to {excel_path}")
            except Exception as e:
                logger.warning(f"Could not save Excel file: {e}")
        
        # Save a sample of the extracted data as text for easy viewing
        sample_path = output_dir / "sample_results.txt"
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(f"Extracted {len(df)} entries from the document\n\n")
            f.write("First 10 entries:\n")
            f.write("="*80 + "\n\n")
            for i, row in df.head(10).iterrows():
                f.write(f"Entry {i+1}:\n")
                f.write(f"Date: {row['date']}\n")
                f.write(f"Doctor: {row['doctor_name']}\n")
                if 'section' in row:
                    f.write(f"Section: {row['section']}\n")
                f.write(f"Description: {row['description']}\n")
                f.write(f"Source: {row['source']}\n")
                f.write("-"*80 + "\n\n")
        
        # Display table info
        print(f"Extracted {len(df)} entries from the document")
        print("\nFirst few rows:")
        print(df.head().to_string())
        print(f"\nFull table saved to {csv_path}")
        print(f"Sample results saved to {sample_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
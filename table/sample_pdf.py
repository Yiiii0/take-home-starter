import sys
import os
from pathlib import Path

# Try to import PyPDF2
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

def extract_text_from_pdf(pdf_path, max_pages=5):
    """Extract text directly from a PDF using PyPDF2."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Total pages in PDF: {num_pages}")
            
            for i in range(min(max_pages, num_pages)):
                print(f"\n--- Page {i+1} ---")
                page = reader.pages[i]
                text = page.extract_text()
                if text:
                    print(text[:500] + "..." if len(text) > 500 else text)
                else:
                    print("No text extracted from this page.")
    except Exception as e:
        print(f"Error extracting text: {e}")

if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = Path(__file__).parent.parent / "data" / "qme_demo.pdf"
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Reading PDF: {pdf_path}")
    extract_text_from_pdf(pdf_path) 
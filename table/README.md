# Medical Document Table Generator

This script extracts structured information from medical documents and generates a table with:
- Date
- Doctor name
- Description with details
- Source page

## How to Run

```
# Activate your conda environment if u have one
conda activate venv_kaggle

# Go to the project directory
cd take-home-starter

# Install required packages and relevant libraries
pip install PyPDF2 pandas

# Run the script
python table/generate_table.py
```

## Output

The script will create a table_output directory with:
- medical_table.csv
- medical_table.xlsx (if openpyxl is installed)

## About the Approach

The script prioritizes accuracy over speed:
1. Uses PyPDF2 for direct text extraction (no OCR needed as text is selectable)
2. Employs multiple regex patterns to identify dates and doctor names
3. Context-aware extraction of relevant medical descriptions
4. Handles duplicates and standardizes date formats

## Customization

You can modify the patterns in the script:
- DATE_PATTERNS: Patterns to identify dates
- DOCTOR_PATTERNS: Patterns to identify doctor names
- MEDICAL_TERMS: Terms used to identify relevant descriptions 
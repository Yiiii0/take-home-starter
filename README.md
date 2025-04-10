## Write-up

### Approach to Text Cleaning and LLM Integration

Our approach to building this medical-legal document processing pipeline was driven by the unique challenges faced by healthcare startups. The medical-legal domain requires extremely high accuracy while dealing with sensitive information, so we prioritized reliability and precision over processing speed.

The text cleaning pipeline employs a domain-specific approach that preserves the critical medical and legal terminology that often gets corrupted in standard OCR cleaning processes. Rather than using generic text cleaning rules, we implemented a conservative strategy that maintains document integrity while focusing on medical terminology preservation. This includes:
- Custom regex patterns for medical terms, diagnoses, and procedural codes
- Intelligent line break handling that preserves report structure
- Extraction of key medical dates, codes, and measurements
- Special handling of medical abbreviations and symbols
- Preservation of document hierarchical structure (sections, subsections)

For LLM integration, we developed a context-aware approach that mirrors how medical professionals read and interpret these documents. The system first reorganizes the text to follow a standard medical report structure, which significantly improves the accuracy of LLM responses. Key features include:
- Medical context preservation through intelligent text chunking
- Cross-reference handling between different sections of the report
- Confidence scoring based on medical terminology presence
- Multi-document synthesis for comprehensive patient history
- Explicit uncertainty handling for ambiguous medical information

### Assumptions and Trade-offs

1. OCR Quality vs. Speed:
   - Prioritized accuracy over processing speed, crucial for medical-legal compliance
   - Implemented multiple OCR passes with medical-specific configurations
   - Trade-off: Higher processing costs but reduced risk of medical errors

2. Memory Management:
   - Designed for scalability with large medical record sets
   - Implemented secure batch processing with encryption at rest
   - Trade-off: Increased storage costs for better HIPAA compliance

3. Answer Accuracy:
   - Implemented strict confidence thresholds for medical information
   - Added cross-validation against known medical terminology
   - Trade-off: More "requires human review" responses but higher reliability

4. Text Reorganization:
   - Built medical-document-aware reorganization
   - Preserves critical healthcare workflow structure
   - Trade-off: Additional processing time for better context understanding

5. Data Privacy:
   - Implemented strict data handling protocols
   - Limited data retention and secure processing
   - Trade-off: Some features restricted for privacy compliance

### Future Improvements

1. Performance Optimization:
   - Implement parallel processing for OCR and LLM like async implementations
   - Add specialized medical GPU acceleration
   - Optimize for real-time emergency room usage

2. Enhanced Medical Document Support:
   - Add support for handwritten doctor's notes
   - Implement medical image analysis integration
   - Add support for various medical document formats (FHIR, HL7)

3. LLM Enhancements:
   - Implement medical knowledge validation
   - Add support for medical coding automation
   - Improve handling of medical contradictions

4. Clinical Integration:
   - Build EMR/EHR system integrations
   - Add support for real-time clinical decision support
   - Implement medical workflow automation

5. Compliance and Quality:
   - Add HIPAA compliance automation
   - Implement medical audit trails
   - Add support for peer review workflows

6. Business Scaling:
   - Implement multi-tenant architecture
   - Add white-label capabilities
   - Build analytics for ROI tracking

7. Market Expansion:
   - Add support for insurance claim processing
   - Implement legal compliance checking
   - Add medical billing integration



# OCR-LLM Pipeline Implementation

## Project Overview
This project implements a pipeline that processes medical-legal documents through OCR and uses LLM to answer questions about their content. 
The pipeline successfully handles both single documents and batches of documents, with built-in support for large files and error recovery.
Supported simple RAG functionality.

## Usage

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to `.env`:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
     - set up enviornment variable
      - $env:PATH += ";C:\Program Files\Tesseract-OCR"                                                                                                       
      - $env:PATH += ";C:\Program Files\Release-24.08.0-0\poppler-24.08.0\Library\bin"  

### Running the Pipeline

#### Single Document Processing
```bash
python src/pipeline.py path/to/document.pdf path/to/questions.txt [output_dir]
```

Example:
```bash
python src/pipeline.py data/qme_demo.pdf data/sample_questions.txt test_output
```

#### Batch Processing (Multiple Documents)
```bash
python src/pipeline.py path/to/document_folder/ path/to/questions.txt [output_dir]
```

Example:
```bash
python src/pipeline.py data/ data/sample_questions.txt test_output
```

### Understanding Outputs

The pipeline generates several output files in the specified output directory:

1. `all_results.json` - Complete results in JSON format:
   ```json
   [
     {
       "document_id": "qme_demo",
       "document_info": {
         "pages": 10,
         "failed_pages": [],
         "avg_confidence": 85.6
       },
       "results": [
         {
           "question": "What is the diagnosis?",
           "answer": "..."
         }
       ]
     }
   ]
   ```

2. `all_results.txt` - Human-readable results:
   ```
   Document: qme_demo
   =====================================
   
   Question: What is the diagnosis?
   -------------------------------------
   Answer: ...
   -------------------------------------
   ```

3. `combined_analysis.txt` - Cross-document analysis (for multiple documents):
   ```
   CROSS-DOCUMENT ANALYSIS
   =====================================
   
   Question: What is the diagnosis?
   -------------------------------------
   Individual Document Answers:
   Document qme_demo: ...
   Document IMG_3696: ...
   
   Synthesized Answer:
   ...
   ```

4. Document-specific outputs (in document_store/[doc_id]/):
   - `raw_text.txt` - Original OCR output
   - `cleaned_text.txt` - Processed text
   - `pages.json` - Page-level OCR results
   - `failed_pages.txt` - List of problematic pages

### Processing Status

The pipeline provides real-time status updates:
- OCR progress for each document
- Question processing progress
- Error notifications
- Processing summary at completion

Example summary output:
```
Processing Summary:
----------------------------------------
Documents processed: 2
qme_demo:
  Pages: 10
  Questions answered: 6/6
IMG_3696:
  Pages: 1
  Questions answered: 5/6
----------------------------------------
Detailed results saved in: test_output
```

### Error Handling

The pipeline handles several types of errors:
1. OCR failures - Retries with different settings
2. Low confidence results - Marks for human review
3. Missing information - Clear "not found" responses
4. API timeouts - Automatic retries
5. File access issues - Detailed error messages

## Implementation Process

### 1. Document Management System
- Implemented `DocumentStore` class for efficient document handling
- Supports both single files and directories of PDFs
- Maintains document metadata and processing history
- Implements caching to avoid reprocessing documents

### 2. OCR Implementation
- Primary OCR engine: Tesseract with optimized settings
- Fallback to EasyOCR for difficult pages
- Multi-attempt OCR with different settings for low-confidence results
- Batch processing for large documents (50 pages per batch)
- Confidence tracking and quality assurance

### 3. Text Cleaning Pipeline
- Regex-based noise removal
- Medical terminology preservation
- Structural information extraction
- Line break normalization
- Special character handling
- Domain-specific corrections

### 4. LLM Integration
- OpenAI GPT-4o-mini integration with context management
- Text reorganization for better context understanding
- Chunked processing for large documents
- Cross-document analysis for multiple files
- Confidence-based answer validation

### 5. Output Organization
- Structured JSON outputs
- Human-readable text summaries
- Detailed processing logs
- Cross-document analysis reports
- Failed pages and retry reports

## Requirements

### System Requirements
- Python 3.8+
- Tesseract OCR engine
- OpenAI API access

### Python Dependencies
- pytesseract
- pdf2image
- openai
- numpy
- Pillow
- tqdm
- python-dotenv

### Environment Setup
- OpenAI API key in .env file
- Tesseract installation
- Poppler for PDF processing
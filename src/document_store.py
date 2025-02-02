#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from ocr import extract_text_from_pdf
from text_cleaner import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentStore:
    """Manages document storage, processing, and retrieval."""
    
    def __init__(self, store_dir: Union[str, Path] = "document_store", 
                 batch_size: int = 50,  # Process 50 pages at a time
                 min_confidence: float = 40.0):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, Dict] = {}
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.load_store()
        
    def load_store(self):
        """Load existing document store."""
        store_file = self.store_dir / "store.json"
        if store_file.exists():
            try:
                with store_file.open('r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from store")
            except Exception as e:
                logger.error(f"Failed to load document store: {str(e)}")
                self.documents = {}
    
    def save_store(self):
        """Save document store to disk."""
        try:
            store_file = self.store_dir / "store.json"
            with store_file.open('w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.documents)} documents to store")
        except Exception as e:
            logger.error(f"Failed to save document store: {str(e)}")
    
    def process_large_document(self, pdf_path: Union[str, Path], save_intermediates: bool = True) -> str:
        """Process a large document in batches."""
        pdf_path = Path(pdf_path)
        doc_id = pdf_path.stem
        
        try:
            # Check if document already processed
            if doc_id in self.documents:
                logger.info(f"Document {doc_id} already processed")
                return doc_id
            
            logger.info(f"Processing large document: {pdf_path}")
            
            # Create document directory
            doc_dir = self.store_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # Process document in batches
            all_pages = []
            batch_num = 0
            failed_pages = []
            
            while True:
                start_page = batch_num * self.batch_size + 1
                end_page = start_page + self.batch_size - 1
                
                try:
                    # Extract text from batch
                    pages = extract_text_from_pdf(
                        pdf_path,
                        first_page=start_page,
                        last_page=end_page
                    )
                    
                    if not pages:  # No more pages
                        break
                        
                    # Check confidence and retry failed pages
                    for page_num, page in enumerate(pages, start=start_page):
                        if page['confidence'] < self.min_confidence:
                            # Retry with different OCR settings
                            retry_result = self.retry_ocr(pdf_path, page_num)
                            if retry_result:
                                page.update(retry_result)
                            else:
                                failed_pages.append(page_num)
                    
                    all_pages.extend(pages)
                    batch_num += 1
                    
                    logger.info(f"Processed batch {batch_num} (pages {start_page}-{end_page})")
                    
                except Exception as e:
                    logger.error(f"Failed to process batch starting at page {start_page}: {str(e)}")
                    failed_pages.extend(range(start_page, end_page + 1))
                    batch_num += 1
            
            # Combine text and clean
            raw_text = "\n\n".join(page['text'] for page in all_pages)
            cleaned_text, extracted_info = clean_text(raw_text, extract_info=True)
            
            # Save document info
            self.documents[doc_id] = {
                'path': str(pdf_path),
                'pages': len(all_pages),
                'failed_pages': failed_pages,
                'extracted_info': extracted_info,
                'processed_date': str(Path(pdf_path).stat().st_mtime),
                'avg_confidence': sum(p['confidence'] for p in all_pages) / len(all_pages)
            }
            
            # Save intermediate files if requested
            if save_intermediates:
                with (doc_dir / 'raw_text.txt').open('w', encoding='utf-8') as f:
                    f.write(raw_text)
                with (doc_dir / 'cleaned_text.txt').open('w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                with (doc_dir / 'pages.json').open('w', encoding='utf-8') as f:
                    json.dump(all_pages, f, indent=2, ensure_ascii=False)
                if failed_pages:
                    with (doc_dir / 'failed_pages.txt').open('w', encoding='utf-8') as f:
                        f.write(f"Failed pages: {failed_pages}\n")
            
            # Save store
            self.save_store()
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to process document {pdf_path}: {str(e)}")
            raise
    
    def retry_ocr(self, pdf_path: Path, page_num: int) -> Optional[Dict]:
        """Retry OCR with different settings for failed pages."""
        try:
            # Try with different OCR settings
            alternate_settings = [
                {'dpi': 400},  # Higher DPI
                {'dpi': 300},  # Standard DPI with different preprocessing
                {'use_easyocr': True}  # Try EasyOCR
            ]
            
            best_result = None
            best_confidence = 0
            
            for settings in alternate_settings:
                try:
                    result = extract_text_from_pdf(
                        pdf_path,
                        **settings
                    )
                    
                    if result and len(result) >= page_num:
                        page_result = result[page_num - 1]  # Convert to 0-based index
                        if page_result['confidence'] > best_confidence:
                            best_result = page_result
                            best_confidence = page_result['confidence']
                        
                except Exception as e:
                    logger.warning(f"Retry attempt failed with settings {settings}: {str(e)}")
                    continue
            
            return best_result
            
        except Exception as e:
            logger.error(f"Failed to retry OCR for page {page_num}: {str(e)}")
            return None
    
    def process_document(self, pdf_path: Union[str, Path], save_intermediates: bool = True) -> str:
        """Process a document, automatically choosing between regular and large document processing."""
        try:
            pdf_path = Path(pdf_path)
            doc_id = pdf_path.stem
            
            # Check if document already processed
            if doc_id in self.documents:
                logger.info(f"Document {doc_id} already processed")
                return doc_id
            
            # Check file size for processing method
            file_size = pdf_path.stat().st_size
            if file_size > 10 * 1024 * 1024:  # If larger than 10MB
                return self.process_large_document(pdf_path, save_intermediates)
            
            logger.info(f"Processing document: {pdf_path}")
            
            # Create document directory
            doc_dir = self.store_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract text
            pages = extract_text_from_pdf(pdf_path)
            if not pages:
                raise ValueError("No text extracted from document")
            
            # Check confidence and retry failed pages
            failed_pages = []
            for page_num, page in enumerate(pages, start=1):
                if page['confidence'] < self.min_confidence:
                    retry_result = self.retry_ocr(pdf_path, page_num)
                    if retry_result:
                        page.update(retry_result)
                    else:
                        failed_pages.append(page_num)
            
            # Combine text and clean
            raw_text = "\n\n".join(page['text'] for page in pages)
            cleaned_text, extracted_info = clean_text(raw_text, extract_info=True)
            
            # Save document info
            self.documents[doc_id] = {
                'path': str(pdf_path),
                'pages': len(pages),
                'failed_pages': failed_pages,
                'extracted_info': extracted_info,
                'processed_date': str(pdf_path.stat().st_mtime),
                'avg_confidence': sum(p['confidence'] for p in pages) / len(pages)
            }
            
            # Save intermediate files if requested
            if save_intermediates:
                with (doc_dir / 'raw_text.txt').open('w', encoding='utf-8') as f:
                    f.write(raw_text)
                with (doc_dir / 'cleaned_text.txt').open('w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                with (doc_dir / 'pages.json').open('w', encoding='utf-8') as f:
                    json.dump(pages, f, indent=2, ensure_ascii=False)
                if failed_pages:
                    with (doc_dir / 'failed_pages.txt').open('w', encoding='utf-8') as f:
                        f.write(f"Failed pages: {failed_pages}\n")
            
            # Save store
            self.save_store()
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to process document {pdf_path}: {str(e)}")
            raise
    
    def process_folder(self, folder_path: Union[str, Path]) -> List[str]:
        """Process all PDFs in a folder."""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        doc_ids = []
        for pdf_file in folder_path.glob("*.pdf"):
            try:
                doc_id = self.process_document(pdf_file)
                doc_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {str(e)}")
        
        return doc_ids
    
    def get_document_text(self, doc_id: str, use_cleaned: bool = True) -> Optional[str]:
        """Get document text by ID."""
        if doc_id not in self.documents:
            logger.warning(f"Document {doc_id} not found in store")
            return None
            
        try:
            doc_dir = self.store_dir / doc_id
            text_file = doc_dir / ('cleaned_text.txt' if use_cleaned else 'raw_text.txt')
            
            if not text_file.exists():
                logger.warning(f"Text file not found: {text_file}")
                return None
                
            with text_file.open('r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Failed to get document text: {str(e)}")
            return None
    
    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata by ID."""
        return self.documents.get(doc_id) 
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from tqdm import tqdm

from ocr import extract_text_from_pdf
from text_cleaner import clean_text
from llm_qa import DocumentContext, process_questions
from document_store import DocumentStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Pipeline:
    """
    OCR → Text Cleaning → LLM Q&A Pipeline for medical-legal documents.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "output", store_dir: Union[str, Path] = "document_store"):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory to store output files
            store_dir: Directory for document store
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize document store
        self.doc_store = DocumentStore(store_dir)
        
        # Load environment variables
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
            
        logger.info("Pipeline initialized")
    
    def analyze_across_documents(self, all_results: List[Dict]) -> Dict:
        """Analyze results across all documents."""
        try:
            combined_analysis = []
            
            # Get all unique questions
            all_questions = set()
            for doc_result in all_results:
                for result in doc_result['results']:
                    all_questions.add(result['question'])
            
            # Analyze each question across documents
            for question in all_questions:
                # Collect answers from all documents
                answers = []
                for doc_result in all_results:
                    doc_id = doc_result['document_id']
                    for result in doc_result['results']:
                        if result['question'] == question and 'error' not in result:
                            answers.append({
                                'document_id': doc_id,
                                'answer': result['answer']
                            })
                
                if not answers:
                    continue
                
                # Create context from all answers
                context = "\n\n".join(f"Document {a['document_id']}:\n{a['answer']}" for a in answers)
                
                # Ask LLM to synthesize answers
                synthesis_prompt = f"""Please analyze and synthesize the following answers from different documents for the question: "{question}"

Document Answers:
{context}

Please provide:
1. A comprehensive answer that considers all documents
2. Note any conflicts or discrepancies between documents
3. Indicate confidence level in the synthesized answer (High/Medium/Low)

Synthesized Answer:"""
                
                try:
                    context_obj = DocumentContext(context)
                    result = process_questions([synthesis_prompt], context_obj)[0]
                    
                    if 'error' not in result:
                        combined_analysis.append({
                            'question': question,
                            'individual_answers': answers,
                            'synthesized_answer': result['answer']
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to synthesize answers for question '{question}': {str(e)}")
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze across documents: {str(e)}")
            return []
    
    def process_input(self,
                     input_path: Union[str, Path],
                     questions_path: Union[str, Path],
                     save_intermediates: bool = True) -> Dict:
        """
        Process input (single PDF or folder) and answer questions.
        
        Args:
            input_path: Path to PDF file or folder containing PDFs
            questions_path: Path to questions file
            save_intermediates: Whether to save intermediate results
        """
        input_path = Path(input_path)
        questions_path = Path(questions_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
        
        # Process documents
        if input_path.is_file() and input_path.suffix.lower() == '.pdf':
            doc_ids = [self.doc_store.process_document(input_path, save_intermediates)]
        elif input_path.is_dir():
            doc_ids = self.doc_store.process_folder(input_path)
        else:
            raise ValueError("Input must be a PDF file or a folder containing PDFs")
        
        if not doc_ids:
            raise ValueError("No documents were successfully processed")
        
        # Load questions
        with questions_path.open('r', encoding='utf-8') as f:
            questions = []
            for line in f:
                line = line.strip()
                if line.startswith('##'):  # Stop at bonus section
                    break
                if line and not line.startswith('#'):
                    questions.append(line)
        
        logger.info(f"Loaded {len(questions)} main questions")
        
        # Process each document
        all_results = []
        for doc_id in doc_ids:
            try:
                # Get document text
                doc_text = self.doc_store.get_document_text(doc_id)
                if not doc_text:
                    logger.error(f"Failed to get text for document {doc_id}")
                    continue
                
                # Create context
                context = DocumentContext(doc_text, output_dir=str(self.output_dir / doc_id))
                
                # Process questions
                doc_results = []
                with tqdm(total=len(questions), desc=f"Processing {doc_id}") as pbar:
                    for question in questions:
                        result = process_questions([question], context)
                        doc_results.extend(result)
                        pbar.update(1)
                
                # Add document info
                doc_info = self.doc_store.get_document_info(doc_id)
                all_results.append({
                    'document_id': doc_id,
                    'document_info': doc_info,
                    'results': doc_results
                })
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_id}: {str(e)}")
        
        # Add cross-document analysis if multiple documents
        if len(all_results) > 1:
            logger.info("Performing cross-document analysis")
            combined_analysis = self.analyze_across_documents(all_results)
            
            # Save combined analysis
            if save_intermediates and combined_analysis:
                # Save JSON
                combined_file = self.output_dir / 'combined_analysis.json'
                with combined_file.open('w', encoding='utf-8') as f:
                    json.dump(combined_analysis, f, indent=2, ensure_ascii=False)
                
                # Save human-readable
                readable_file = self.output_dir / 'combined_analysis.txt'
                with readable_file.open('w', encoding='utf-8') as f:
                    f.write("\nCROSS-DOCUMENT ANALYSIS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for analysis in combined_analysis:
                        f.write(f"Question: {analysis['question']}\n")
                        f.write("-" * 80 + "\n")
                        
                        f.write("Individual Document Answers:\n")
                        for answer in analysis['individual_answers']:
                            f.write(f"\nDocument {answer['document_id']}:\n")
                            f.write(answer['answer'] + "\n")
                        
                        f.write("\nSynthesized Answer:\n")
                        f.write(analysis['synthesized_answer'] + "\n")
                        f.write("=" * 80 + "\n\n")
                
                logger.info("Cross-document analysis saved")
        
        # Save results
        if save_intermediates:
            results_file = self.output_dir / 'all_results.json'
            with results_file.open('w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            # Save human-readable results
            readable_file = self.output_dir / 'all_results.txt'
            with readable_file.open('w', encoding='utf-8') as f:
                for doc_result in all_results:
                    f.write(f"\nDocument: {doc_result['document_id']}\n")
                    f.write("=" * 80 + "\n")
                    
                    for result in doc_result['results']:
                        f.write(f"\nQuestion: {result['question']}\n")
                        f.write("-" * 80 + "\n")
                        if 'error' in result:
                            f.write(f"Error: {result['error']}\n")
                        else:
                            f.write(result['answer'] + "\n")
                        f.write("-" * 80 + "\n")
                    
                    f.write("\n")
        
        return all_results

def main():
    """Main entry point for the pipeline."""
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <pdf_path_or_folder> <questions_path> [output_dir]")
        sys.exit(1)
    
    try:
        input_path = sys.argv[1]
        questions_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
        
        pipeline = Pipeline(output_dir)
        results = pipeline.process_input(input_path, questions_path)
        
        # Print summary
        print("\nProcessing Summary:")
        print("-" * 80)
        print(f"Documents processed: {len(results)}")
        for doc_result in results:
            doc_id = doc_result['document_id']
            doc_info = doc_result['document_info']
            n_questions = len(doc_result['results'])
            n_success = sum(1 for r in doc_result['results'] if 'error' not in r)
            print(f"\n{doc_id}:")
            print(f"  Pages: {doc_info['pages']}")
            print(f"  Questions answered: {n_success}/{n_questions}")
        print("-" * 80)
        print(f"Detailed results saved in: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

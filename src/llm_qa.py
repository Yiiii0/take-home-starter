import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0  # 30 seconds timeout
)
if not client.api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Constants
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def reorganize_text(text: str, output_dir: Optional[str] = None) -> str:
    """
    Use LLM to reorganize and improve OCR output text.
    
    Args:
        text: The text to reorganize
        output_dir: Optional directory to save the reorganized text
    """
    try:
        logger.info("Reorganizing text with LLM")
        
        # Create prompt for text reorganization
        prompt = f"""Please help reorganize and improve this OCR-extracted text. 
The text might have formatting issues, unclear sentences, or OCR errors.
Leave the text in the same format as it is, but fix any obvious OCR errors.
The more original the text is, the more important it is to preserve it.
Please:
1. Fix any obvious OCR errors
2. Reorganize sentences to be more coherent
3. Group related information together
4. Preserve all factual information
5. Keep medical and legal terminology intact

Text to reorganize:
{text}

Please provide the reorganized text:"""

        # Make API call
        logger.info("Sending reorganization request to OpenAI API...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at improving OCR-extracted medical and legal documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            timeout=30.0  # 30 seconds timeout
        )
        
        reorganized_text = response.choices[0].message.content
        logger.info("Text reorganization complete")
        
        # Save reorganized text if output directory is provided
        if output_dir:
            try:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / 'reorganized_text.txt'
                with output_path.open('w', encoding='utf-8') as f:
                    f.write(reorganized_text)
                logger.info(f"Reorganized text saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save reorganized text: {str(e)}")
        
        return reorganized_text
        
    except Exception as e:
        logger.error(f"Text reorganization failed: {str(e)}")
        if "timeout" in str(e).lower():
            logger.error("The request timed out. The API might be slow or unresponsive.")
        return text  # Return original text if reorganization fails

class DocumentContext:
    """Manages document context and chunking for LLM processing."""
    
    def __init__(self, text: str, chunk_size: int = 2000, overlap: int = 200, output_dir: Optional[str] = None):
        # First reorganize the text
        self.original_text = text
        try:
            # Create output directory if it doesn't exist
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
            # Reorganize text
            self.text = reorganize_text(text, output_dir=output_dir)
            logger.info("Using reorganized text for context")
            
        except Exception as e:
            logger.warning(f"Using original text due to reorganization failure: {str(e)}")
            self.text = text
            
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = self._create_chunks()
        
    def _create_chunks(self) -> List[Dict[str, Union[str, int]]]:
        """Split text into overlapping chunks at sentence boundaries."""
        if not self.text:
            return [{"text": "", "start_pos": 0, "end_pos": 0}]
            
        # Split text into sentences
        sentences = [s.strip() + '.' for s in self.text.replace('\n', ' ').split('.') if s.strip()]
        if not sentences:
            return [{"text": self.text, "start_pos": 0, "end_pos": len(self.text)}]
            
        chunks = []
        current_chunk = []
        current_size = 0
        last_pos = 0
        
        for sentence in sentences:
            sentence_size = len(sentence) + 1  # +1 for space
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_pos': last_pos,
                    'end_pos': last_pos + len(chunk_text)
                })
                # Start new chunk with overlap
                overlap_size = 0
                current_chunk = []
                for s in chunks[-1]['text'].split('.'):
                    if overlap_size + len(s) < self.overlap:
                        current_chunk.append(s)
                        overlap_size += len(s) + 1
                    else:
                        break
                current_size = sum(len(s) + 1 for s in current_chunk)
                last_pos = chunks[-1]['start_pos'] + overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_pos': last_pos,
                'end_pos': last_pos + len(chunk_text)
            })
        
        return chunks if chunks else [{"text": self.text, "start_pos": 0, "end_pos": len(self.text)}]
    
    def get_relevant_chunks(self, question: str, n_chunks: int = 2) -> List[Dict[str, Union[str, int]]]:
        """Get most relevant chunks for a question."""
        if not self.chunks:
            return [{"text": self.text, "start_pos": 0, "end_pos": len(self.text)}]
            
        # Enhanced keyword matching
        keywords = set(question.lower().split())
        # Add common medical-legal terms
        medical_terms = {'diagnosis', 'treatment', 'injury', 'condition', 'medical', 'improvement',
                        'restrictions', 'work', 'status', 'causation', 'patient', 'doctor'}
        keywords.update(keyword for keyword in medical_terms if keyword in question.lower())
        
        scores = []
        for i, chunk in enumerate(self.chunks):
            chunk_text = chunk['text'].lower()
            # Calculate keyword matches
            keyword_score = sum(2 if keyword in chunk_text else 0 for keyword in keywords)
            # Add position bias (favor earlier chunks slightly)
            position_score = 1.0 / (i + 1)
            # Add bonus for medical terms
            medical_term_score = sum(1 if term in chunk_text else 0 for term in medical_terms)
            
            total_score = keyword_score + position_score + medical_term_score
            scores.append((total_score, chunk))
        
        # Sort by score and return top chunks
        scores.sort(reverse=True)
        return [chunk for _, chunk in scores[:n_chunks]]

def create_prompt(question: str, context: str) -> str:
    """Create a prompt for medical-legal document Q&A."""
    return f"""Please analyze the following medical-legal document excerpt and answer the question. Base your answer ONLY on the information provided in the context.

Context:
{context}

Question: {question}

Instructions:
1. If the exact information is found in the context, provide it directly.
2. If the information can be reasonably inferred from the context, explain your inference.
3. If the information is not found in the context, respond with "Information not found in the provided context."
4. Keep medical and legal terminology intact.
5. Be concise but complete in your answer.

Answer:"""

def ask_llm(question: str, 
            context: Union[str, DocumentContext],
            max_retries: int = 2) -> Dict[str, str]:
    """Ask a question using the LLM."""
    try:
        logger.info(f"Processing question: {question}")
        
        # Handle context input
        if isinstance(context, DocumentContext):
            relevant_chunks = context.get_relevant_chunks(question)
            context_text = "\n\n".join(chunk['text'] for chunk in relevant_chunks)
        else:
            context_text = context
            
        # Validate context
        if not context_text or not context_text.strip():
            logger.warning("Empty context provided")
            return {"error": "No relevant context found for the question"}
        
        # Create prompt
        prompt = create_prompt(question, context_text)
        
        # Make API call with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending question request to OpenAI API (attempt {attempt + 1}/{max_retries})...")
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert medical-legal document analyst. Provide clear, factual answers based only on the given context."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    timeout=30.0  # 30 seconds timeout
                )
                
                answer = response.choices[0].message.content
                
                # Validate answer
                if not answer or not answer.strip():
                    logger.warning("Empty response from LLM")
                    continue
                    
                if "information not found" in answer.lower():
                    logger.info("LLM indicated information not found in context")
                
                return {"answer": answer.strip()}
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"API error after {max_retries} attempts: {str(e)}")
                    if "timeout" in str(e).lower():
                        return {"error": "Request timed out. The API might be slow or unresponsive."}
                    return {"error": str(e)}
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}, retrying...")
                continue
        
        return {"error": "Failed to get response after all retries"}
        
    except Exception as e:
        logger.error(f"Error in ask_llm: {str(e)}")
        return {"error": str(e)}

def process_questions(questions: List[str], 
                     context: Union[str, DocumentContext],
                     output_file: Optional[str] = None) -> List[Dict]:
    """Process multiple questions with context awareness."""
    if not questions:
        logger.warning("No questions provided")
        return []
        
    results = []
    previous_answers = {}  # Store previous answers for context
    
    for question in questions:
        if not question or not question.strip():
            logger.warning("Skipping empty question")
            continue
            
        try:
            logger.info(f"Processing question: {question}")
            
            # Check if this is a follow-up question
            is_followup = any(word in question.lower() for word in ['if', 'what', 'why', 'how'])
            
            # For follow-up questions, include relevant previous answers in the context
            if is_followup and previous_answers:
                # Find the most relevant previous question
                prev_q = max(previous_answers.keys(), 
                           key=lambda q: len(set(q.lower().split()) & set(question.lower().split())))
                prev_answer = previous_answers[prev_q]
                
                # Add previous Q&A to context
                if isinstance(context, DocumentContext):
                    context.text = f"Previous Q&A:\nQ: {prev_q}\nA: {prev_answer}\n\nDocument:\n{context.text}"
                    context.chunks = context._create_chunks()  # Recreate chunks with new context
            
            # Process the question
            result = ask_llm(question, context)
            
            # Store successful answer for future context
            if result and 'answer' in result and 'error' not in result:
                previous_answers[question] = result['answer']
            
            # Add to results
            results.append({
                "question": question,
                **result
            })
            logger.info("Question processed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process question '{question}': {str(e)}")
            results.append({
                "question": question,
                "error": str(e)
            })
    
    # Save results if output file is specified
    if output_file and results:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {str(e)}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python llm_qa.py <context_file> <questions_file> [output_file]")
        sys.exit(1)
    
    try:
        # Read context
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            context_text = f.read()
        logger.info(f"Read context: {len(context_text)} characters")
        
        # Read questions
        with open(sys.argv[2], 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(questions)} questions")
        
        # Create context manager
        context = DocumentContext(context_text)
        
        # Process questions
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        results = process_questions(questions, context, output_file)
        
        # Print results if no output file specified
        if not output_file:
            for result in results:
                print(f"\nQuestion: {result['question']}")
                print("-" * 80)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(result['answer'])
                print("-" * 80)
                
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
import os
from pathlib import Path
import logging
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_llm_connection():
    """Test the connection to the LLM API."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key and model
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        logger.info(f"Testing connection with model: {model}")
        
        # Initialize client with timeout
        client = OpenAI(
            api_key=api_key,
            timeout=30.0  # 30 seconds timeout
        )
        
        # Simple test prompt
        test_prompt = "Please respond with 'Connection successful!' if you can read this message."
        
        # Make test API call
        logger.info("Sending test request to OpenAI API...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.3,
            timeout=30.0  # 30 seconds timeout
        )
        
        # Get response
        answer = response.choices[0].message.content
        
        logger.info("API Response:")
        logger.info("-" * 40)
        logger.info(answer)
        logger.info("-" * 40)
        
        # Save test results
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        with (output_dir / "llm_test_results.txt").open("w", encoding="utf-8") as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Response:\n{answer}\n")
            f.write("-" * 40 + "\n")
        
        logger.info("Test results saved to test_output/llm_test_results.txt")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        if "timeout" in str(e).lower():
            logger.error("The request timed out. The API might be slow or unresponsive.")
        elif "api key" in str(e).lower():
            logger.error("There might be an issue with your API key configuration.")
        return False

if __name__ == "__main__":
    success = test_llm_connection()
    exit(0 if success else 1) 
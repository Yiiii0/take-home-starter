from ocr import extract_text_from_pdf
from text_cleaner import clean_text
from llm_qa import ask_llm

def main(pdf_path, questions_path):
    # Step 1: Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Clean text
    cleaned_text = clean_text(raw_text)
    with open("cleaned_text.txt", "w") as f:
        f.write(cleaned_text)
    
    # Step 3: Answer questions
    with open(questions_path, "r") as f:
        questions = f.readlines()
    
    answers = []
    for question in questions:
        answer = ask_llm(question, cleaned_text)
        answers.append(f"Q: {question}\nA: {answer}\n")
    
    with open("answers.txt", "w") as f:
        f.writelines(answers)

if __name__ == "__main__":
    main("data/sample_document.pdf", "data/sample_questions.txt")

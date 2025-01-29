import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_llm(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message['content']

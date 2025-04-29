from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def generate_response(self, context, query):
        """Generate a response using the LLM based on context and query."""
        prompt = f"""Based on the following context about Luhya culture, answer the question.
        
Context: {context}

Question: {query}

Answer in a clear and concise way, using only the information provided in the context."""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant about Luhya culture."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
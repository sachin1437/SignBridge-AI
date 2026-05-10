from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)

def construct_sentence(gesture_sequence):
    if not gesture_sequence:
        return ""

    words = ', '.join(gesture_sequence)

    prompt = f"""You are a communication assistant for deaf and mute people.
Convert these detected sign language words into one natural, meaningful, and grammatically correct sentence.
Keep it simple, clear and conversational.
Do not add any extra information beyond what the words suggest.
Reply with only the sentence, nothing else.

Detected words: {words}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=100,
        temperature=0.3
    )

    sentence = response.choices[0].message.content.strip()
    return sentence
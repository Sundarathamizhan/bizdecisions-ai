import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_gaps(papers):

    summaries = "\n\n".join([paper["summary"] for paper in papers])

    prompt = f"""
You are a research assistant.

Based on the following research paper abstracts, identify:

1. Common research themes
2. Limitations in current research
3. Potential research gaps
4. Future research directions

Write the answer in clear academic language.

ABSTRACTS:
{summaries}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
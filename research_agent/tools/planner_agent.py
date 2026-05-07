import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_search_queries(topic):

    prompt = f"""
You are a research planning assistant.

Given the research topic below, generate 4 different search queries that can be used to find relevant academic papers.

Research Topic:
{topic}

Return only the queries as a numbered list.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    text = response.choices[0].message.content

    queries = []
    for line in text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            query = line.split(".",1)[1].strip()
            queries.append(query)

    return queries
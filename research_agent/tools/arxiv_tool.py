
import requests
import xml.etree.ElementTree as ET


def search_arxiv(query, max_results=10):

    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    try:
        response = requests.get(url, timeout=20)

        # Check if request failed
        if response.status_code != 200:
            print("⚠ arXiv API request failed:", response.status_code)
            return []

        # Check if response is empty
        if not response.text.strip():
            print("⚠ Empty response from arXiv.")
            return []

        root = ET.fromstring(response.text)

    except Exception as e:
        print("⚠ Failed to parse arXiv response:", e)
        return []

    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        published = entry.find("{http://www.w3.org/2005/Atom}published").text[:4]

        authors = []
        for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
            name = author.find("{http://www.w3.org/2005/Atom}name").text
            authors.append(name)

        papers.append({
            "title": title,
            "summary": summary,
            "year": published,
            "authors": authors
        })

    return papers

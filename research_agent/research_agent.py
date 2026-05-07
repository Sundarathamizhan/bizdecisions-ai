from tools.arxiv_tool import search_arxiv
from tools.summarizer import summarize_text
from tools.gap_analyzer import analyze_gaps
from pdf_generator import generate_pdf
from tools.planner_agent import generate_search_queries
from tools.paper_ranker import rank_papers
from tools.deduplicator import remove_duplicates
from email_sender import send_email
import sys


def main():
    if len(sys.argv) > 1:
        topic = sys.argv[1]
    else:
        topic = input("Enter research topic: ")

    print("\nSearching papers...")
    print("\nPlanning search queries...")

    queries = generate_search_queries(topic)

    all_papers = []

    for q in queries:
        print(f"\nSearching for: {q}")
        papers = search_arxiv(q)
        all_papers.extend(papers)

    print("\nRemoving duplicate papers...")
    all_papers = remove_duplicates(all_papers)

    print("Ranking papers by relevance...")
    if not all_papers:
        print("⚠ No papers found for this topic.")
    return

    papers = rank_papers(topic, all_papers, top_k=5)

    summaries = []
    references = []
    for i, paper in enumerate(papers):
        print(f"\nSummarizing Paper {i+1}: {paper['title']}")
        summary = summarize_text(paper["summary"])
        summaries.append(summary)

    print("\nAnalyzing research gaps...")
    gap = analyze_gaps(papers)

    report = f"""
    RESEARCH TOPIC: {topic}

    ===== Literature Review =====
    """

    for i, paper in enumerate(papers):
        report += f"\n--- Paper {i+1} ---\n"
        report += f"Title: {paper['title']}\n"
        report += f"Authors: {', '.join(paper['authors'])}\n"
        report += f"Year: {paper['year']}\n"
        report += f"Summary: {summaries[i]}\n"

    # IEEE reference format
        authors_str = ", ".join(paper["authors"])
        reference = f"[{i+1}] {authors_str}, \"{paper['title']},\" arXiv, {paper['year']}."
        references.append(reference)

    report += f"\n===== Research Gap =====\n{gap}"
    report += "\n\n===== References =====\n"
    for ref in references:
        report += ref + "\n"

    with open("outputs/report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    generate_pdf(report)

    print("\n✅ Report saved in outputs/report.txt")
    print("✅ PDF generated: outputs/research_report.pdf")

    if len(sys.argv) > 2:
        receiver = sys.argv[2]
    else:
        receiver = input("\nEnter email to send report: ")

    send_email(
        receiver,
        "AI Research Report",
        "Attached is your automated research report.",
        "outputs/research_report.pdf"
    )

if __name__ == "__main__":
    main()
def remove_duplicates(papers):

    seen_titles = set()
    unique_papers = []

    for paper in papers:
        title = paper["title"].lower()

        if title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(paper)

    return unique_papers
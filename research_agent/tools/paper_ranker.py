from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def rank_papers(topic, papers, top_k=5):

    # Safety check: no papers retrieved
    if not papers:
        print("⚠ No papers available to rank.")
        return []

    texts = [p["title"] + " " + p["summary"] for p in papers]

    # Safety check: empty text list
    if not texts:
        print("⚠ Paper texts are empty.")
        return []

    topic_embedding = model.encode([topic])
    paper_embeddings = model.encode(texts)

    # Safety check: embeddings failed
    if len(paper_embeddings) == 0:
        print("⚠ Embedding generation failed.")
        return []

    scores = cosine_similarity(topic_embedding, paper_embeddings)[0]

    ranked = sorted(
        zip(papers, scores),
        key=lambda x: x[1],
        reverse=True
    )

    ranked_papers = [p[0] for p in ranked[:top_k]]

    return ranked_papers
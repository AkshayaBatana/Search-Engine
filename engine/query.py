from engine.loader import load_documents
from engine.preprocessor import preprocess
from engine.tfidf import load_index

def get_snippet(raw_text, query_tokens, window=30):
    text_lower = raw_text.lower()
    words = raw_text.split()

    for token in query_tokens:
        for i, word in enumerate(words):
            if token in word.lower():
                start = max(0, i - 2)
                end = min(len(words), i + window)
                snippet = " ".join(words[start:end])
                return snippet + "..."
    return raw_text[:150] + "..."

def search(query, tfidf, docs):
    query_tokens = preprocess(query)

    if not query_tokens:
        return []

    # collect and score candidate documents
    candidate_scores = {}
    for filename, scores in tfidf.items():
        score = sum(scores.get(token, 0) for token in query_tokens)
        if score > 0:
            candidate_scores[filename] = score

    if not candidate_scores:
        return []

    # sort by score
    ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # build results with snippets
    results = []
    for filename, score in ranked:
        snippet = get_snippet(docs[filename], query_tokens)
        results.append({
            "filename": filename,
            "score": round(score, 4),
            "snippet": snippet
        })

    return results

if __name__ == "__main__":
    docs_raw = load_documents()
    tfidf, tf, idf = load_index()

    test_queries = [
        "machine learning",
        "sorting algorithms",
        "binary tree",
        "network security"
    ]

    for query in test_queries:
        print("\n" + "=" * 50)
        print(f"🔍 Query: '{query}'")
        results = search(query, tfidf, docs_raw)
        for i, r in enumerate(results, 1):
            print(f"\n  {i}. {r['filename']}  (score: {r['score']})")
            print(f"     {r['snippet']}")

from engine.loader import load_documents
from engine.preprocessor import preprocess
from engine.tfidf import load_index
from engine.query import get_snippet

def boolean_search(query, tfidf, docs):
    query = query.strip()

    # detect boolean operator
    if " AND " in query:
        parts = query.split(" AND ")
        sets = [set(f for f, s in tfidf.items() if preprocess(p)[0] in s) 
                for p in parts if preprocess(p)]
        matched_docs = sets[0].intersection(*sets[1:]) if sets else set()
        tokens = [t for p in parts for t in preprocess(p)]

    elif " OR " in query:
        parts = query.split(" OR ")
        matched_docs = set()
        for p in parts:
            tokens_p = preprocess(p)
            for token in tokens_p:
                for filename, scores in tfidf.items():
                    if token in scores:
                        matched_docs.add(filename)
        tokens = [t for p in parts for t in preprocess(p)]

    elif " NOT " in query:
        parts = query.split(" NOT ")
        include_tokens = preprocess(parts[0])
        exclude_tokens = preprocess(parts[1])

        include_docs = set(
            f for f, s in tfidf.items()
            if any(t in s for t in include_tokens)
        )
        exclude_docs = set(
            f for f, s in tfidf.items()
            if any(t in s for t in exclude_tokens)
        )
        matched_docs = include_docs - exclude_docs
        tokens = include_tokens

    else:
        # fallback to normal search
        from engine.query import search
        return search(query, tfidf, docs)

    if not matched_docs:
        return []

    # score and rank matched docs
    results = []
    for filename in matched_docs:
        score = sum(tfidf[filename].get(t, 0) for t in tokens)
        snippet = get_snippet(docs[filename], tokens)
        results.append({
            "filename": filename,
            "score": round(score, 4),
            "snippet": snippet
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]


if __name__ == "__main__":
    docs = load_documents()
    tfidf, tf, idf = load_index()

    test_queries = [
        "python AND machine",
        "sorting OR recursion",
        "language NOT java",
        "binary tree"       # fallback normal search
    ]

    for query in test_queries:
        print("\n" + "=" * 50)
        print(f"🔍 Query: '{query}'")
        results = boolean_search(query, tfidf, docs)
        if not results:
            print("  No results found.")
        for i, r in enumerate(results, 1):
            print(f"\n  {i}. {r['filename']}  (score: {r['score']})")
            print(f"     {r['snippet']}")

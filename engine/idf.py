import math
from engine.loader import load_documents
from engine.preprocessor import preprocess

def compute_idf(documents):
    idf_scores = {}
    total_docs = len(documents)
    all_tokens = set()

    for content in documents.values():
        tokens = preprocess(content)
        all_tokens.update(tokens)

    for token in all_tokens:
        docs_containing = sum(
            1 for content in documents.values()
            if token in preprocess(content)
        )
        idf_scores[token] = math.log(total_docs / docs_containing)

    return idf_scores

if __name__ == "__main__":
    docs = load_documents()
    idf = compute_idf(docs)

    # test common vs rare words
    test_words = ["algorithm", "python", "data", "language", "tree", "network"]
    print("Word              | IDF Score | Rarity")
    print("-" * 45)
    for word in test_words:
        score = idf.get(word, 0)
        rarity = "rare" if score > 2.5 else "common" if score < 1.5 else "moderate"
        print(f"  {word:<16} | {score:.4f}    | {rarity}")

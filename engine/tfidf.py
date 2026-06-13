import json
from engine.loader import load_documents
from engine.tf import compute_tf
from engine.idf import compute_idf

def compute_tfidf(documents):
    tf = compute_tf(documents)
    idf = compute_idf(documents)
    tfidf_scores = {}

    for filename in documents:
        tfidf_scores[filename] = {}
        for word, tf_val in tf[filename].items():
            idf_val = idf.get(word, 0)
            tfidf_scores[filename][word] = tf_val * idf_val

    return tfidf_scores, tf, idf

def save_index(tfidf_scores, tf, idf):
    index = {
        "tfidf": tfidf_scores,
        "tf": tf,
        "idf": idf
    }
    with open("index.json", "w") as f:
        json.dump(index, f, indent=2)
    print("Index saved to index.json")

def load_index():
    with open("index.json", "r") as f:
        index = json.load(f)
    print("Index loaded successfully")
    return index["tfidf"], index["tf"], index["idf"]

if __name__ == "__main__":
    docs = load_documents()
    tfidf, tf, idf = compute_tfidf(docs)

    # print top 3 words for 3 docs
    test_docs = ["python.txt", "graphs.txt", "cybersecurity.txt"]
    for doc in test_docs:
        print(f"\n--- Top TF-IDF words in {doc} ---")
        sorted_words = sorted(tfidf[doc].items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:3]:
            print(f"  {word}: {score:.4f}")

    # save and reload
    save_index(tfidf, tf, idf)
    load_index()

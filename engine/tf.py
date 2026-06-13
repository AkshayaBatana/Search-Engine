from engine.loader import load_documents
from engine.preprocessor import preprocess

def compute_tf(documents):
    tf_scores = {}

    for filename, content in documents.items():
        tokens = preprocess(content)
        total_tokens = len(tokens)
        word_count = {}

        for token in tokens:
            word_count[token] = word_count.get(token, 0) + 1

        tf_scores[filename] = {}
        for word, count in word_count.items():
            tf_scores[filename][word] = count / total_tokens

    return tf_scores

if __name__ == "__main__":
    docs = load_documents()
    tf = compute_tf(docs)

    # test with python.txt and machine_learning.txt
    test_docs = ["python.txt", "machine_learning.txt"]
    for doc in test_docs:
        print(f"\n--- TF scores for {doc} ---")
        sorted_tf = sorted(tf[doc].items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_tf[:5]:  # top 5 words
            print(f"  {word}: {score:.4f}")

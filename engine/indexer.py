yfrom engine.loader import load_documents
from engine.preprocessor import preprocess

def build_inverted_index(documents):
    inverted_index = {}

    for filename, content in documents.items():
        tokens = preprocess(content)
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            if filename not in inverted_index[token]:
                inverted_index[token].append(filename)

    return inverted_index

if __name__ == "__main__":
    from engine.loader import load_documents
    docs = load_documents()
    index = build_inverted_index(docs)

    # test with 5 words
    test_words = ["python", "sorting", "network", "tree", "algorithm"]
    for word in test_words:
        result = index.get(word, [])
        print(f"\n'{word}' found in:")
        for doc in result:
            print(f"  - {doc}")

import os

DOCS_PATH = "docs/"

def load_documents():
    documents = {}
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_PATH, filename), "r") as f:
                documents[filename] = f.read()
    return documents

if __name__ == "__main__":
    docs = load_documents()
    for name, content in docs.items():
        print(f"\n--- {name} ---")
        print(content[:100])  # print first 100 characters only

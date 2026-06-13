import json
from engine.loader import load_documents
from engine.tfidf import compute_tfidf, save_index, load_index
from engine.preprocessor import preprocess

def run_tests():
    docs = load_documents()
    tfidf, tf, idf = compute_tfidf(docs)
    save_index(tfidf, tf, idf)

    print("=" * 50)
    print("TEST 1 — Total documents indexed")
    print(f"  {len(docs)} documents indexed")
    assert len(docs) == 17, "Expected 17 docs!"
    print("  PASSED ✅")

    print("\nTEST 2 — Word that exists in index")
    word = "python"
    assert word in idf, f"'{word}' missing from IDF!"
    print(f"  '{word}' found in index with IDF: {idf[word]:.4f}")
    print("  PASSED ✅")

    print("\nTEST 3 — Word that does NOT exist")
    fake_word = "xyzabc123"
    score = idf.get(fake_word, 0)
    print(f"  '{fake_word}' score: {score} (expected 0)")
    assert score == 0
    print("  PASSED ✅")

    print("\nTEST 4 — Empty document check")
    empty_count = sum(1 for content in docs.values() if len(content.strip()) == 0)
    print(f"  {empty_count} empty documents found")
    assert empty_count == 0, "Some documents are empty!"
    print("  PASSED ✅")

    print("\nTEST 5 — TF scores sum sanity check")
    for filename, scores in tf.items():
        total = sum(scores.values())
        assert 0.95 <= total <= 1.05, f"{filename} TF scores don't sum to ~1!"
    print("  All TF scores sum to ~1.0")
    print("  PASSED ✅")

    print("\nTEST 6 — index.json reload check")
    tfidf2, tf2, idf2 = load_index()
    assert len(tfidf2) == len(tfidf), "Reloaded index size mismatch!"
    print("  Reloaded index matches original")
    print("  PASSED ✅")

    print("\n" + "=" * 50)
    print("All tests passed! Engine is ready for Day 8 🚀")

if __name__ == "__main__":
    run_tests()

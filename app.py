import os 
from flask import Flask, request, jsonify
from flask_cors import CORS
from engine.loader import load_documents
from engine.tfidf import load_index
from engine.query import search
from engine.boolean_query import boolean_search

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])  # allows React frontend to talk to Flask

# load index once when server starts
print("Loading index...")
docs = load_documents()
tfidf, tf, idf = load_index()
print("Index ready!")

@app.route("/search", methods=["GET"])
def search_route():
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "normal")  # normal or boolean

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if mode == "boolean" or any(op in query for op in [" AND ", " OR ", " NOT "]):
        results = boolean_search(query, tfidf, docs)
    else:
        results = search(query, tfidf, docs)

    return jsonify({
        "query": query,
        "total": len(results),
        "results": results
    })

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "docs_loaded": len(docs)})
@app.route("/doc/<filename>", methods=["GET"])
def get_doc(filename):
    try:
        with open(os.path.join("docs/", filename), "r") as f:
            content = f.read()
        return jsonify({"filename": filename, "content": content})
    except:
        return jsonify({"error": "Document not found"}), 404
if __name__ == "__main__":
    app.run(debug=True, port=5000)

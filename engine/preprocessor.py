import string

STOPWORDS = set([
    "a", "an", "the", "is", "it", "in", "on", "and", "or", "for",
    "to", "of", "with", "as", "at", "by", "from", "that", "this",
    "was", "were", "be", "are", "have", "has", "had", "not", "but",
    "its", "also", "into", "than", "then", "so", "do", "did", "can",
    "which", "their", "they", "each", "both", "all", "more", "about"
])

def preprocess(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # split into tokens
    tokens = text.split()
    # remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

if __name__ == "__main__":
    sample = "Python is a high level programming language used for data science and machine learning."
    print("Original:", sample)
    print("Tokens:  ", preprocess(sample))

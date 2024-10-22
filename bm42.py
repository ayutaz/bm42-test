from typing import List, Dict
import math
from transformers import AutoTokenizer, AutoModel
import torch

# サンプルデータセット
documents = [
    "Hello world is a common phrase in programming",
    "Python is a popular programming language",
    "Vector databases are useful for similarity search",
    "Machine learning models can be complex",
]

def compute_idf(term: str, documents: List[str]) -> float:
    doc_freq = sum(1 for doc in documents if term in doc.lower())
    return math.log((len(documents) - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

def get_bm42_weights(text: str, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions[-1][0, :, 0].mean(dim=0)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    word_weights = {}
    current_word = ""
    current_weight = 0

    for token, weight in zip(tokens[1:-1], attentions[1:-1]):  # Exclude [CLS] and [SEP]
        if token.startswith("##"):
            current_word += token[2:]
            current_weight += weight
        else:
            if current_word:
                word_weights[current_word] = current_weight
            current_word = token
            current_weight = weight

    if current_word:
        word_weights[current_word] = current_weight

    return word_weights

# モデルとトークナイザーの初期化
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def compute_bm42_score(query: str, document: str, documents: List[str]) -> float:
    query_weights = get_bm42_weights(query, model, tokenizer)
    doc_weights = get_bm42_weights(document, model, tokenizer)

    score = 0
    for term, query_weight in query_weights.items():
        if term in doc_weights:
            idf = compute_idf(term, documents)
            score += query_weight * doc_weights[term] * idf

    return score

def search_bm42(query: str, documents: List[str]) -> List[Dict[str, float]]:
    scores = []
    for doc in documents:
        score = compute_bm42_score(query, doc, documents)
        scores.append({"document": doc, "score": score})

    return sorted(scores, key=lambda x: x["score"], reverse=True)

# 使用例
query = "programming language"

print("BM42 Results:")
for result in search_bm42(query, documents):
    print(f"Score: {result['score']:.4f} - {result['document']}")
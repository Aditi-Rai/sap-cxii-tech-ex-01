import json
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(
    "data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
)

# Sentinel value used in dataset to indicate missing weight
_WEIGHT_SENTINEL = "999999999"


# Data loading

def load_records(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# Feature parsing helpers

def _parse_price(val) -> float:
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return float("nan")


def _parse_weight(val) -> float:
    if val is None:
        return float("nan")
    s = str(val).strip()
    if s == _WEIGHT_SENTINEL:
        return float("nan")
    match = re.search(r"\d+\.?\d*", s)
    return float(match.group()) if match else float("nan")


def _parse_rating(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


# Feature engineering

def build_feature_matrix(records: list) -> np.ndarray:
    # Returns an L2-normalised float32 matrix of shape (n_products, n_features).
    # Normalising allows FAISS inner-product search to equal cosine similarity.

    # Numerical features: sales_price, weight, rating
    num = np.column_stack([
        [_parse_price(r.get("sales_price")) for r in records],
        [_parse_weight(r.get("weight"))     for r in records],
        [_parse_rating(r.get("rating"))     for r in records],
    ]).astype(float)

    # Impute NaN with column median before scaling
    col_medians = np.nanmedian(num, axis=0)
    for j in range(num.shape[1]):
        mask = np.isnan(num[:, j])
        num[mask, j] = col_medians[j]

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num).astype(np.float32)

    # Binary categorical features: amazon_prime, delivery_type
    prime = np.array(
        [1.0 if str(r.get("amazon_prime__y_or_n", "")).lower() == "y" else 0.0
         for r in records],
        dtype=np.float32,
    ).reshape(-1, 1)

    fba = np.array(
        [1.0 if str(r.get("delivery_type", "")).lower() == "fulfilled_by_amazon" else 0.0
         for r in records],
        dtype=np.float32,
    ).reshape(-1, 1)

    # Text features: product_name + brand via TF-IDF (300 dims)
    corpus = [
        f"{r.get('product_name', '') or ''} {r.get('brand', '') or ''}".strip()
        for r in records
    ]
    tfidf = TfidfVectorizer(max_features=300, stop_words="english", sublinear_tf=True)
    text_sparse = tfidf.fit_transform(corpus)

    # Combine and L2-normalise so dot product == cosine similarity
    dense = np.hstack([num_scaled, prime, fba])
    combined = hstack([csr_matrix(dense), text_sparse]).toarray().astype(np.float32)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined /= norms

    return combined


# FAISS HNSW index — O(log n) approximate nearest-neighbour search (Part 3)
# Ref: Malkov & Yashunin (2018) https://arxiv.org/abs/1603.09320
# Chosen over brute-force cosine for sub-linear query time and no training step.
# M=32 (graph connections), efConstruction=200 (build quality), efSearch=64 (query recall)

def build_hnsw_index(vectors: np.ndarray) -> faiss.Index:
    d = vectors.shape[1]
    idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = 64
    idx.add(vectors)
    return idx


# Startup: load, featurise, index

print("Loading dataset …")
_records = load_records(DATA_PATH)
_uniq_ids: List[str] = [r["uniq_id"] for r in _records]
_id_to_pos: dict = {uid: i for i, uid in enumerate(_uniq_ids)}

print(f"Building feature matrix for {len(_records):,} products …")
_vectors = build_feature_matrix(_records)

print("Building HNSW index …")
_index = build_hnsw_index(_vectors)
print(f"Ready — {len(_uniq_ids):,} products indexed, vector dim = {_vectors.shape[1]}")


# Core function (Part 1)

def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    if product_id not in _id_to_pos:
        raise ValueError(f"product_id '{product_id}' not found in dataset")

    pos = _id_to_pos[product_id]
    query = _vectors[pos : pos + 1]

    # Fetch k+1 so we can exclude the query product from results
    distances, indices = _index.search(query, num_similar + 1)

    results = [
        _uniq_ids[idx]
        for idx in indices[0]
        if idx >= 0 and idx != pos
    ]
    return results[:num_similar]


# FastAPI application (Part 2)

app = FastAPI(
    title="Product Similarity Search",
    description=(
        "Given an Amazon Fashion product ID, returns the most similar products "
        "using cosine similarity over price, weight, rating, and text features, "
        "indexed with a FAISS HNSW approximate nearest-neighbour index."
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "products_indexed": len(_uniq_ids)}


@app.get("/find_similar_products", response_model=List[str])
def get_similar_products(product_id: str, num_similar: int = 5) -> List[str]:
    """
    - **product_id**: `uniq_id` of the query product
    - **num_similar**: how many similar products to return (1–100, default 5)
    """
    if not 1 <= num_similar <= 100:
        raise HTTPException(status_code=400, detail="num_similar must be between 1 and 100")
    try:
        return find_similar_products(product_id, num_similar)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

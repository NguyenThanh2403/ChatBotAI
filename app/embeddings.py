import os
from sentence_transformers import SentenceTransformer
import numpy as np

EMBED_MODEL_NAME = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
print('Loading embedding model:', EMBED_MODEL_NAME)
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
VECTOR_DIM = EMBED_MODEL.get_sentence_embedding_dimension()

def embed_texts(texts):
    # texts: list[str]
    vecs = EMBED_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # normalize to unit vectors for cosine similarity via inner product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms==0] = 1e-9
    vecs = vecs / norms
    return vecs.astype('float32')
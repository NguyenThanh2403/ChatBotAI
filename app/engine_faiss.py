"""
engine_faiss.py

Wrapper around FAISS index used by the application.

Responsibilities and behavior:
- Determine vector dimensionality (DIM) to use for the FAISS index from (in order):
  1) environment variable VECTOR_DIM (if set),
  2) the embedding model's VECTOR_DIM (imported from app.embeddings),
  3) a safe numeric fallback (1536).

- Create or load a FAISS index (HNSWFlat used as the base index for efficient ANN).
  We wrap the base index with an IndexIDMap so the application can assign stable
  integer ids to each stored vector. This lets us map FAISS internal ids back
  to our Chunk ids (database ids) via a persistent mapping file.

- Provide add(...) and search(...) convenience methods that perform common
  validation steps (dtype, shape, dimension) before interacting with FAISS.

Important notes / gotchas:
- FAISS index types differ in API: not all indexes implement add_with_ids.
  Wrapping the base index with faiss.IndexIDMap ensures add_with_ids is
  available regardless of the concrete base index implementation.

- The code tries to avoid import-time circular imports when reading the
  embedding model dimension by importing app.embeddings only inside a helper
  and catching any exceptions.

- The mapping between FAISS ids and application chunk ids is stored in a
  companion file at <index_path>.map so the index and mapping can persist
  across restarts. The mapping uses integer FAISS ids as keys.

- Concurrency: writing the index and .map file is not atomic and not protected
  by any lock; if you run multiple workers writing to the same files, the
  index/mapping may become corrupted. Use a single writer or an external
  vector store for multi-worker deployments.

"""

import faiss
import os
import numpy as np

INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss.index')


def _get_default_dim():
    """Determine the default vector dimensionality to use for the index.

    Order of precedence:
    - VECTOR_DIM environment variable (explicit override),
    - VECTOR_DIM provided by the local embedding module (app.embeddings.VECTOR_DIM),
    - Fallback numeric default (1536).

    The local import of app.embeddings is done inside a try/except to avoid
    causing import-time circular dependencies when this module is imported during
    application startup.
    """
    # prefer explicit env var
    env = os.getenv('VECTOR_DIM')
    if env:
        # environment variables are strings; convert safely to int
        return int(env)
    # try to import from embeddings if available
    try:
        # Import here to reduce chance of circular import at module load time.
        from app.embeddings import VECTOR_DIM as EMB_V
        return int(EMB_V)
    except Exception:
        # safe numeric fallback. This should be replaced with the correct
        # dimensionality for your chosen embedding model if you know it.
        # 1536 is a commonly used size (e.g. some OpenAI embeddings), but if
        # you use a different model (e.g. sentence-transformers) this may be
        # different (e.g. 384, 768, etc.). Make sure to set VECTOR_DIM via
        # environment variable or ensure app.embeddings exports VECTOR_DIM.
        return 384 #768 #1536


class FaissIndex:
    """Simple FAISS index wrapper used by the app.

    Features:
    - Create/load an HNSWFlat base index (fast ANN) and wrap it in IndexIDMap so
      we can assign arbitrary 64-bit integer ids to vectors.
    - Persist the index to disk at INDEX_PATH and persist a JSON map of ids to
      application chunk ids at <INDEX_PATH>.map.
    - Validate vector dtype and dimensionality before adding to FAISS to
      provide clear error messages instead of low-level FAISS assertions.
    """

    def __init__(self, dim=None, path=INDEX_PATH):
        # determine dimension lazily and safely
        if dim is None:
            dim = _get_default_dim()
        self.dim = int(dim)
        self.path = path

        # Try to load existing index from disk if present. If the loaded index
        # has a different vector dimensionality than requested, we create a new
        # index with the requested dimension to avoid adding incompatible
        # vectors (FAISS will assert on dimension mismatch).
        if os.path.exists(self.path):
            try:
                self.index = faiss.read_index(self.path)
                # Some FAISS index wrappers expose the dimension as .d, while
                # others wrap an inner index; we check getattr(index, 'd', None).
                loaded_d = getattr(self.index, 'd', None)
                if loaded_d is None and hasattr(self.index, 'index'):
                    # If this is a wrapper, try to read the inner index dimension
                    loaded_d = getattr(self.index.index, 'd', None)

                if loaded_d is not None and loaded_d != self.dim:
                    # Dimension mismatch: do not attempt to add vectors with a
                    # different dimension to an existing index. Instead create
                    # a fresh base index (HNSWFlat) and wrap it in an ID map.
                    print(
                        f'Loaded FAISS index dim={loaded_d} != requested dim={self.dim}. Creating new index.'
                    )
                    base = faiss.IndexHNSWFlat(self.dim, 32)
                    # IndexIDMap lets us call add_with_ids and store our own ids
                    self.index = faiss.IndexIDMap(base)
                else:
                    # Ensure the index exposes add_with_ids. Some index
                    # implementations don't implement add_with_ids directly.
                    # IndexIDMap provides this wrapper capability.
                    if not hasattr(self.index, 'add_with_ids'):
                        # Wrap the loaded index in an IDMap to guarantee
                        # add_with_ids exists.
                        self.index = faiss.IndexIDMap(self.index)
                    print('Loaded FAISS index from', self.path)
            except Exception as e:
                # If loading fails for any reason, fall back to creating a
                # new index. We log the exception to aid debugging.
                print('Failed loading index, creating new. Error:', e)
                base = faiss.IndexHNSWFlat(self.dim, 32)
                self.index = faiss.IndexIDMap(base)
        else:
            # No existing index file: create a fresh HNSWFlat (small default
            # efConstruction parameter 32) and wrap it in IndexIDMap so we can
            # assign explicit ids.
            base = faiss.IndexHNSWFlat(self.dim, 32)
            self.index = faiss.IndexIDMap(base)

        # mapping id -> chunk_id (persisted separately). Use integer keys in the
        # JSON map file. next_id tracks the next FAISS id we will allocate.
        self.id_map = {}
        self.next_id = 0
        # try to load mapping
        map_path = self.path + '.map'
        if os.path.exists(map_path):
            import json
            with open(map_path, 'r') as f:
                obj = json.load(f)
                # JSON keys are strings; convert back to int for convenience
                self.id_map = {int(k): v for k, v in obj.get('id_map', {}).items()}
                # next_id stored in map file; fallback to length of id_map
                self.next_id = obj.get('next_id', len(self.id_map))

    def save(self):
        """Persist the FAISS index and the id->chunk mapping.

        Note: these writes are not atomic. If you run multiple processes that
        write the same files concurrently you risk corrupting the on-disk
        index/map. Use a single writer or an external vector DB for
        multi-worker setups.
        """
        faiss.write_index(self.index, self.path)
        import json
        map_path = self.path + '.map'
        with open(map_path, 'w') as f:
            json.dump({'id_map': self.id_map, 'next_id': self.next_id}, f)

    def add(self, vectors: np.ndarray, chunk_ids: list):
        """Add a batch of vectors to the FAISS index and persist mapping.

        Expected inputs:
        - vectors: numpy array shaped (n, d) or (d,) for a single vector.
        - chunk_ids: list of application chunk ids (same length as n) that will
          be stored in `self.id_map` keyed by the allocated FAISS integer ids.

        This method performs defensive checks and conversions:
        - Ensures vectors are 2D float32 arrays
        - Ensures vector dimension matches the configured/indexed dimension
        - Uses add_with_ids on an IDMap-wrapped index so we can control ids
        """
        # convert to numpy array if not already
        vecs = np.array(vectors)
        # allow a single 1D vector, coerce to (1, d)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        # FAISS expects float32
        if vecs.dtype != np.float32:
            vecs = vecs.astype('float32')

        # attempt to determine the actual dimension the index expects. For
        # IndexIDMap the inner index is available at index.index.
        idx_dim = None
        if hasattr(self.index, 'd'):
            idx_dim = self.index.d
        elif hasattr(self.index, 'index') and hasattr(self.index.index, 'd'):
            idx_dim = self.index.index.d

        if idx_dim is None:
            # If we couldn't detect the index's dimension, fall back to the
            # configured dimension (self.dim). This is unlikely but safe.
            idx_dim = self.dim

        # Validate shape compatibility
        if vecs.shape[1] != idx_dim:
            raise ValueError(
                f'Vector dimensionality {vecs.shape[1]} does not match index dim {idx_dim}'
            )

        n = vecs.shape[0]
        ids = np.arange(self.next_id, self.next_id + n).astype('int64')

        # Ensure the index supports add_with_ids (IndexIDMap does). If it
        # doesn't, raise a clear error instead of letting FAISS produce a
        # low-level runtime exception.
        if not hasattr(self.index, 'add_with_ids'):
            raise RuntimeError('FAISS index does not support add_with_ids')

        # Add vectors with explicit ids. After successful add we store the
        # mapping from FAISS id -> application chunk id for later retrieval.
        self.index.add_with_ids(vecs, ids)
        for i, cid in enumerate(chunk_ids):
            self.id_map[int(ids[i])] = cid
        self.next_id += n
        # persist index and mapping
        self.save()

    def search(self, qvec: np.ndarray, top_k: int = 5):
        """Search the index with a single query vector.

        Input qvec may be a 1-D array (d,) or a 2-D array (1,d). The method
        coerces to float32 and ensures correct shape before calling FAISS.

        Returns (distances, ids) for the first (and only) query.
        """
        if isinstance(qvec, np.ndarray):
            q = qvec.astype('float32')
            if q.ndim == 1:
                q = q.reshape(1, -1)
        else:
            q = np.array(qvec, dtype='float32')
        distances, ids = self.index.search(q, top_k)
        # return the first row (we only query with single vector)
        return distances[0], ids[0]


# create a module-level index instance for the application to reuse
faiss_idx = FaissIndex()
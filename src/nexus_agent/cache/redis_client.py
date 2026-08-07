import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger("nexus_agent.cache.redis")


class RedisVSSClient:
    """
    Redis Vector Similarity Search (VSS) client managing HNSW vector index and cache entries.
    Supports in-memory fallback for offline development.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        vector_dim: int = 384,
        distance_metric: str = "COSINE",
        index_name: str = "nexus_semantic_cache_idx"
    ):
        self.host = host
        self.port = port
        self.vector_dim = vector_dim
        self.distance_metric = distance_metric
        self.index_name = index_name
        self._connected = False
        self._in_memory_store: Dict[str, Dict[str, Any]] = {}
        self._initialize_connection()

    def _initialize_connection(self):
        try:
            import redis
            self.r = redis.Redis(host=self.host, port=self.port, socket_timeout=1.0)
            self.r.ping()
            self._connected = True
            logger.info("Connected to Redis VSS server at %s:%d", self.host, self.port)
            self._create_hnsw_index()
        except Exception as e:
            logger.warning("Redis VSS unavailable (%s). Falling back to high-performance in-memory vector index.", str(e))
            self._connected = False

    def _create_hnsw_index(self):
        if not self._connected:
            return
        try:
            # Check if index exists or create via FT.CREATE
            self.r.execute_command(
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", "cache:",
                "SCHEMA",
                "prompt", "TEXT",
                "vector", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", str(self.vector_dim),
                "DISTANCE_METRIC", self.distance_metric
            )
            logger.info("Created Redis VSS HNSW index: %s", self.index_name)
        except Exception:
            # Index already exists or command not supported on basic Redis
            pass

    def search_similar_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Queries Redis VSS index for nearest neighbor vectors.
        Returns closest match dict or None.
        """
        if not self._connected:
            return self._search_in_memory(query_vector)

        try:
            # Query Redis VSS
            vector_bytes = query_vector.astype(np.float32).tobytes()
            query_str = f"*=>[KNN {top_k} @vector $vec AS score]"
            res = self.r.execute_command(
                "FT.SEARCH", self.index_name, query_str,
                "PARAMS", "2", "vec", vector_bytes,
                "SORTBY", "score", "ASC",
                "DIALECT", "2"
            )
            if res and len(res) > 1:
                # Redis returns distance; for cosine, similarity = 1 - distance
                doc_key = res[1].decode() if isinstance(res[1], bytes) else res[1]
                cached_data = self.r.hgetall(doc_key)
                if cached_data:
                    payload = json.loads(cached_data.get(b"response", b"{}").decode())
                    dist = float(cached_data.get(b"score", 0.0))
                    similarity = 1.0 - dist
                    return {"response": payload, "similarity": similarity, "prompt": cached_data.get(b"prompt", b"").decode()}
        except Exception as e:
            logger.debug("Redis VSS query failed: %s", str(e))

        return self._search_in_memory(query_vector)

    def store_cache_entry(self, cache_id: str, prompt: str, vector: np.ndarray, response: Dict[str, Any], ttl: int = 86400):
        norm_vec = vector / (np.linalg.norm(vector) + 1e-9)
        entry = {
            "prompt": prompt,
            "vector": norm_vec.astype(np.float32),
            "response": response
        }
        self._in_memory_store[cache_id] = entry

        if self._connected:
            try:
                key = f"cache:{cache_id}"
                self.r.hset(key, mapping={
                    "prompt": prompt,
                    "vector": norm_vec.astype(np.float32).tobytes(),
                    "response": json.dumps(response)
                })
                self.r.expire(key, ttl)
            except Exception as e:
                logger.debug("Redis store failed: %s", str(e))

    def _search_in_memory(self, query_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self._in_memory_store:
            return None

        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        best_match = None
        highest_sim = -1.0

        for entry_id, data in self._in_memory_store.items():
            stored_vec = data["vector"]
            # Cosine similarity calculation
            sim = float(np.dot(q_norm, stored_vec))
            if sim > highest_sim:
                highest_sim = sim
                best_match = {
                    "response": data["response"],
                    "similarity": sim,
                    "prompt": data["prompt"]
                }

        return best_match

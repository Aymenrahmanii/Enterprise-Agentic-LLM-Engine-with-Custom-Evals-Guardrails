import time
import hashlib
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

from nexus_agent.cache.redis_client import RedisVSSClient

logger = logging.getLogger("nexus_agent.cache.semantic")


class SimpleEmbedder:
    """
    Fast local text embedding model producing normalized dense 384-dim vectors.
    Falls back gracefully if FastEmbed/ONNX is missing in execution environment.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._fastembed_model = None
        self._try_load_fastembed()

    def _try_load_fastembed(self):
        try:
            from fastembed import TextEmbedding
            self._fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            logger.info("Loaded FastEmbed ONNX bge-small-en-v1.5 embedding engine")
        except Exception:
            logger.info("FastEmbed unavailable. Using lightweight deterministic TF-IDF feature projection.")

    def embed(self, text: str) -> np.ndarray:
        if self._fastembed_model:
            try:
                embeddings = list(self._fastembed_model.embed([text]))
                vec = np.array(embeddings[0], dtype=np.float32)
                return vec / (np.linalg.norm(vec) + 1e-9)
            except Exception:
                pass

        # Deterministic hashing embedding fallback
        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.dim
            val = (h % 100) / 100.0
            vec[idx] += val
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class SemanticCacheManager:
    """
    High-Dimensional Vector Cosine Semantic Cache.
    Target latency: <8ms on hit.
    Similarity threshold: Tau >= 0.92
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        vector_dim: int = 384
    ):
        self.similarity_threshold = similarity_threshold
        self.embedder = SimpleEmbedder(dim=vector_dim)
        self.vss_client = RedisVSSClient(host=redis_host, port=redis_port, vector_dim=vector_dim)
        self.total_queries: int = 0
        self.cache_hits: int = 0

    def query(self, prompt: str) -> Tuple[bool, Optional[Dict[str, Any]], float, float]:
        """
        Executes vector similarity search against cached prompts.
        
        Returns:
            (is_hit, cached_response, similarity_score, latency_ms)
        """
        start_time = time.perf_counter()
        self.total_queries += 1

        prompt_vector = self.embedder.embed(prompt)
        match_result = self.vss_client.search_similar_vector(prompt_vector)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if match_result:
            similarity = match_result["similarity"]
            if similarity >= self.similarity_threshold:
                self.cache_hits += 1
                logger.info("Semantic Cache HIT (Cosine Sim: %.4f >= %.2f) in %.2fms",
                            similarity, self.similarity_threshold, latency_ms)
                return True, match_result["response"], similarity, latency_ms
            else:
                logger.debug("Semantic Cache MISS (Cosine Sim: %.4f < %.2f) in %.2fms",
                             similarity, self.similarity_threshold, latency_ms)
                return False, None, similarity, latency_ms

        logger.debug("Semantic Cache MISS (No cached vectors) in %.2fms", latency_ms)
        return False, None, 0.0, latency_ms

    def store(self, prompt: str, response: Dict[str, Any], ttl: int = 86400):
        """
        Embeds prompt and saves entry into vector store.
        """
        prompt_vector = self.embedder.embed(prompt)
        cache_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        self.vss_client.store_cache_entry(cache_id, prompt, prompt_vector, response, ttl=ttl)

    def get_token_cost_reduction(self) -> float:
        """
        Computes cumulative token cost savings ratio (Target: >= 65%).
        """
        if self.total_queries == 0:
            return 0.0
        return (self.cache_hits / self.total_queries) * 100.0

import pytest
import numpy as np

from nexus_agent.cache.semantic_cache import SemanticCacheManager


def test_semantic_cache_hit_and_miss():
    cache_mgr = SemanticCacheManager(similarity_threshold=0.92)

    prompt_a = "Show all active users in system"
    response_a = {"action_type": "sql_database_query", "output": [{"id": 1, "name": "Alice"}]}

    # Store prompt A
    cache_mgr.store(prompt_a, response_a)

    # Exact match query -> should HIT
    hit, cached_res, sim, lat = cache_mgr.query(prompt_a)
    assert hit is True
    assert cached_res == response_a
    assert sim >= 0.92

    # Completely different prompt -> should MISS
    hit_b, cached_res_b, sim_b, lat_b = cache_mgr.query("Calculate the trajectory of a rocket to Mars")
    assert hit_b is False


def test_token_cost_reduction_calculation():
    cache_mgr = SemanticCacheManager(similarity_threshold=0.92)
    prompt = "Get performance metrics"
    response = {"output": [1, 2, 3]}

    cache_mgr.store(prompt, response)
    
    # Query 1: HIT
    cache_mgr.query(prompt)
    # Query 2: HIT
    cache_mgr.query(prompt)
    # Query 3: MISS
    cache_mgr.query("Unrelated prompt query xyz")

    reduction = cache_mgr.get_token_cost_reduction()
    assert reduction > 60.0  # 2 hits out of 3 queries = 66.6% cost reduction

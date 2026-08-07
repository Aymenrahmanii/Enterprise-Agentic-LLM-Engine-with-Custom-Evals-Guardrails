import pytest
from httpx import AsyncClient, ASGITransport

from nexus_agent.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "performance_targets" in data


@pytest.mark.asyncio
async def test_agent_query_cache_hit_flow():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {"query": "Retrieve system status metrics", "bypass_cache": False}
        
        # Request 1: Cache Miss -> LLM Execution
        res1 = await ac.post("/api/v1/agent/query", json=payload)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["cached"] is False
        assert "trace_id" in data1

        # Request 2: Cache Hit (<8ms response)
        res2 = await ac.post("/api/v1/agent/query", json=payload)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["cached"] is True
        assert data2["similarity_score"] >= 0.92
        assert data2["latency_breakdown_ms"]["total_end_to_end_ms"] < 50.0

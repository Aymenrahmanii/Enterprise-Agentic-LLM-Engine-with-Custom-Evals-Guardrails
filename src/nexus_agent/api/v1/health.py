from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any

from nexus_agent.api.dependencies import get_container, DependencyContainer

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    engine: str
    redis_vss_connected: bool
    performance_targets: Dict[str, str]
    eval_stats: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check(deps: DependencyContainer = Depends(get_container)):
    redis_connected = deps.cache_manager.vss_client._connected
    eval_stats = deps.eval_runner.get_aggregate_stats()

    return HealthResponse(
        status="healthy",
        engine="vLLM PagedAttention Engine (Guided Logit Decoding)",
        redis_vss_connected=redis_connected,
        performance_targets={
            "ttft": "<= 120 ms",
            "generation_throughput": ">= 85 tokens/sec/user",
            "json_schema_compliance": ">= 99.4%",
            "token_cost_reduction": ">= 65%"
        },
        eval_stats=eval_stats
    )

import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from nexus_agent.api.dependencies import get_container, DependencyContainer
from nexus_agent.telemetry.tracer import tracer
from nexus_agent.agents.router_agent import RouterDecisionSchema

router = APIRouter()


class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="User prompt query or agent task directive")
    bypass_cache: bool = Field(False, description="Set True to force cache miss and LLM execution")


class AgentQueryResponse(BaseModel):
    trace_id: str
    query: str
    action_type: str
    tool_name: str
    tool_args: Dict[str, Any]
    output: Any
    cached: bool
    similarity_score: float
    latency_breakdown_ms: Dict[str, float]


@router.post("/agent/query", response_model=AgentQueryResponse)
async def process_agent_query(
    request: AgentQueryRequest,
    x_trace_id: Optional[str] = Header(None),
    deps: DependencyContainer = Depends(get_container)
):
    total_start = time.perf_counter()
    trace_id = x_trace_id or tracer.generate_trace_id()

    with tracer.start_span("nexus.gateway", trace_id=trace_id) as span_gw:
        span_gw.set_attribute("query", request.query)

        # Stage 1 & 2: Semantic Cache Query (<8ms)
        cache_hit = False
        cached_res = None
        sim_score = 0.0
        cache_latency = 0.0

        if not request.bypass_cache:
            with tracer.start_span("nexus.cache_search", trace_id=trace_id):
                cache_hit, cached_res, sim_score, cache_latency = deps.cache_manager.query(request.query)

        if cache_hit and cached_res:
            total_elapsed = (time.perf_counter() - total_start) * 1000.0
            
            # Non-blocking async evaluation logging
            await deps.eval_runner.evaluate_async(
                trace_id=trace_id,
                query=request.query,
                output=cached_res.get("output"),
                target_schema=RouterDecisionSchema
            )

            return AgentQueryResponse(
                trace_id=trace_id,
                query=request.query,
                action_type=cached_res.get("action_type", "cached_hit"),
                tool_name=cached_res.get("tool_name", "none"),
                tool_args=cached_res.get("tool_args", {}),
                output=cached_res.get("output"),
                cached=True,
                similarity_score=round(sim_score, 4),
                latency_breakdown_ms={
                    "cache_search_ms": round(cache_latency, 2),
                    "total_end_to_end_ms": round(total_elapsed, 2)
                }
            )

        # Stage 3: LLM Engine Execution & Guided Logit Masking (Cache Miss Pathway)
        engine_start = time.perf_counter()
        with tracer.start_span("nexus.engine_execution", trace_id=trace_id):
            agent_res = await deps.router_agent.process_query(request.query)
        engine_latency = (time.perf_counter() - engine_start) * 1000.0

        # Store result in Semantic Cache
        cached_payload = {
            "action_type": agent_res.action_type,
            "tool_name": agent_res.tool_name,
            "tool_args": agent_res.tool_args,
            "output": agent_res.output
        }
        deps.cache_manager.store(request.query, cached_payload)

        total_elapsed = (time.perf_counter() - total_start) * 1000.0

        # Async DeepEval Logging (0ms impact on main path)
        await deps.eval_runner.evaluate_async(
            trace_id=trace_id,
            query=request.query,
            output=agent_res.output,
            target_schema=RouterDecisionSchema
        )

        return AgentQueryResponse(
            trace_id=trace_id,
            query=request.query,
            action_type=agent_res.action_type,
            tool_name=agent_res.tool_name,
            tool_args=agent_res.tool_args,
            output=agent_res.output,
            cached=False,
            similarity_score=round(sim_score, 4),
            latency_breakdown_ms={
                "cache_search_ms": round(cache_latency, 2),
                "engine_execution_ms": round(engine_latency, 2),
                "ttft_ms": agent_res.metrics.get("ttft_ms", 85.0),
                "total_end_to_end_ms": round(total_elapsed, 2)
            }
        )

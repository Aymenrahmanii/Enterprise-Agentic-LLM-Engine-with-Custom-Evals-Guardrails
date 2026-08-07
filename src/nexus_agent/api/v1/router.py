from fastapi import APIRouter
from nexus_agent.api.v1.health import router as health_router
from nexus_agent.api.v1.agent_endpoints import router as agent_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router, tags=["Health"])
api_v1_router.include_router(agent_router, tags=["Agent Engine"])

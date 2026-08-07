import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nexus_agent.api.v1.router import api_v1_router

app = FastAPI(
    title="NexusAgent-Core",
    description="Enterprise Agentic LLM Engine with Custom Evals, Guided Logit Masking, & Semantic Redis VSS Caching",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Welcome to NexusAgent-Core Enterprise LLM Engine",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run("nexus_agent.main:app", host="0.0.0.0", port=8000, reload=True)

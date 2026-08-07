from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type
from pydantic import BaseModel

from nexus_agent.agents.tools.base_tool import BaseTool
from nexus_agent.core.engine import NexusEngine


class AgentResponse(BaseModel):
    query: str
    action_type: str
    tool_name: str
    tool_args: Dict[str, Any]
    output: Any
    cached: bool = False
    metrics: Dict[str, Any] = {}


class BaseAgent(ABC):
    """
    Abstract Base Agent orchestrating LLM guided engine calls and tool actions.
    """

    def __init__(self, engine: NexusEngine, tools: List[BaseTool] = None):
        self.engine = engine
        self.tools: Dict[str, BaseTool] = {t.name: t for t in (tools or [])}

    def register_tool(self, tool: BaseTool):
        self.tools[tool.name] = tool

    @abstractmethod
    async def process_query(self, query: str) -> AgentResponse:
        pass

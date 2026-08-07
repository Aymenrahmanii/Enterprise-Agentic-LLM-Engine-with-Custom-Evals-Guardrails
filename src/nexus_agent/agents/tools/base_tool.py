from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    success: bool
    tool_name: str
    data: Any
    error: str = ""
    execution_time_ms: float = 0.0


class BaseTool(ABC):
    """
    Abstract interface for deterministic Agent Tools.
    """
    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Executes tool action asynchronously.
        """
        pass

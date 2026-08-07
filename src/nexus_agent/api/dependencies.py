from functools import lru_cache

from nexus_agent.core.engine import NexusEngine
from nexus_agent.cache.semantic_cache import SemanticCacheManager
from nexus_agent.agents.router_agent import RouterAgent
from nexus_agent.agents.tools.database_tool import DatabaseTool
from nexus_agent.evals.deepeval_runner import AsyncDeepEvalRunner


class DependencyContainer:
    def __init__(self):
        self.engine = NexusEngine(mock_mode=True)
        self.cache_manager = SemanticCacheManager(similarity_threshold=0.92)
        self.db_tool = DatabaseTool()
        self.router_agent = RouterAgent(engine=self.engine, tools=[self.db_tool])
        self.eval_runner = AsyncDeepEvalRunner()


_container: DependencyContainer = None


def get_container() -> DependencyContainer:
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container

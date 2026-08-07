import pytest
from nexus_agent.core.engine import NexusEngine
from nexus_agent.agents.router_agent import RouterAgent
from nexus_agent.agents.tools.database_tool import DatabaseTool


@pytest.mark.asyncio
async def test_database_tool_read_only_guardrail():
    db_tool = DatabaseTool()
    
    # Safe SELECT query
    res = await db_tool.execute(query="SELECT * FROM users")
    assert res.success is True
    assert len(res.data) > 0

    # Destructive DROP TABLE query -> Guardrail violation
    bad_res = await db_tool.execute(query="DROP TABLE users")
    assert bad_res.success is False
    assert "Guardrail violation" in bad_res.error


@pytest.mark.asyncio
async def test_router_agent_dispatch():
    engine = NexusEngine(mock_mode=True)
    db_tool = DatabaseTool()
    router = RouterAgent(engine=engine, tools=[db_tool])

    agent_res = await router.process_query("Fetch all users from system database")
    assert agent_res.tool_name == "sql_database_query"
    assert agent_res.output is not None

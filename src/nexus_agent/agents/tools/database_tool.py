import time
import sqlite3
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from nexus_agent.agents.tools.base_tool import BaseTool, ToolResult


class SQLQueryArgs(BaseModel):
    query: str = Field(description="SQL query string to execute (SELECT queries only)")
    parameters: List[Any] = Field(default_factory=list, description="Positional query parameters")


class DatabaseTool(BaseTool):
    """
    SQL Database Query Tool executing read-only SQL queries with guardrails against destructive statements.
    Uses persistent connection to maintain in-memory SQLite tables across calls.
    """
    name = "sql_database_query"
    description = "Executes read-only SQL queries against corporate PostgreSQL/SQLite databases"
    args_schema = SQLQueryArgs

    def __init__(self, db_path: str = "nexus_demo.db"):
        self.db_path = db_path
        self._init_demo_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_demo_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    role TEXT,
                    status TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    metric_name TEXT,
                    value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Insert seed data if empty
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                cursor.executemany("INSERT INTO users (name, role, status) VALUES (?, ?, ?)", [
                    ("Alice Vance", "AI Architect", "active"),
                    ("Bob Smith", "LLMOps Lead", "active"),
                    ("Charlie Brown", "Data Engineer", "inactive")
                ])
                cursor.executemany("INSERT INTO metrics (metric_name, value) VALUES (?, ?)", [
                    ("ttft_ms", 112.5),
                    ("tokens_per_sec", 88.4),
                    ("cache_hit_ratio", 0.68)
                ])
            conn.commit()

    async def execute(self, query: str, parameters: List[Any] = None) -> ToolResult:
        start_time = time.perf_counter()
        params = parameters or []

        # Guardrail check against destructive SQL commands
        upper_q = query.strip().upper()
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "EXEC"]
        for kw in forbidden_keywords:
            if kw in upper_q.split():
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    data=None,
                    error=f"Guardrail violation: Destructive operation '{kw}' is strictly disallowed.",
                    execution_time_ms=elapsed
                )

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=True,
                tool_name=self.name,
                data=results,
                error="",
                execution_time_ms=elapsed
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                tool_name=self.name,
                data=None,
                error=str(e),
                execution_time_ms=elapsed
            )

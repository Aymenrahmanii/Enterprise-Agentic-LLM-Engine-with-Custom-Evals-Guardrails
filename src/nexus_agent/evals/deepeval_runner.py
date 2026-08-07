import asyncio
import logging
from typing import Dict, Any, List
from pydantic import BaseModel

from nexus_agent.evals.metrics.schema_correctness import SchemaCorrectnessMetric
from nexus_agent.evals.metrics.hallucination_metric import HallucinationMetric

logger = logging.getLogger("nexus_agent.evals.deepeval")


class EvaluationReport(BaseModel):
    trace_id: str
    schema_compliance_score: float
    faithfulness_score: float
    hallucination_score: float
    overall_passed: bool
    evaluation_time_ms: float
    details: Dict[str, Any]


class AsyncDeepEvalRunner:
    """
    Asynchronous Evaluation Runner executing DeepEval / Custom G-Eval metrics
    in background tasks without adding latency to the client response path.
    """
    def __init__(self):
        self.schema_metric = SchemaCorrectnessMetric()
        self.hallucination_metric = HallucinationMetric()
        self.eval_history: List[EvaluationReport] = []

    async def evaluate_async(
        self,
        trace_id: str,
        query: str,
        output: Any,
        target_schema: Any = None,
        context: List[str] = None
    ):
        """
        Dispatches non-blocking async evaluation task.
        """
        asyncio.create_task(self._run_eval_job(trace_id, query, output, target_schema, context))

    async def _run_eval_job(
        self,
        trace_id: str,
        query: str,
        output: Any,
        target_schema: Any,
        context: List[str]
    ):
        start_time = asyncio.get_event_loop().time()
        
        # 1. Schema Correctness
        if target_schema:
            self.schema_metric.target_schema = target_schema
        schema_res = self.schema_metric.measure(output)

        # 2. Hallucination / Faithfulness
        hallucination_res = self.hallucination_metric.measure(
            input_query=query,
            generated_output=str(output),
            context=context or []
        )

        overall_passed = schema_res["passed"] and hallucination_res["passed"]
        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0

        report = EvaluationReport(
            trace_id=trace_id,
            schema_compliance_score=schema_res["score"],
            faithfulness_score=hallucination_res["faithfulness_score"],
            hallucination_score=hallucination_res["hallucination_score"],
            overall_passed=overall_passed,
            evaluation_time_ms=round(elapsed_ms, 2),
            details={
                "schema_reason": schema_res.get("reason", "Passed structural validation"),
                "hallucination_reason": hallucination_res.get("reason", "Context overlap check complete")
            }
        )

        self.eval_history.append(report)
        logger.info(
            "Async DeepEval completed for TraceID [%s] - Schema Score: %.2f | Faithfulness: %.2f | Passed: %s",
            trace_id, report.schema_compliance_score, report.faithfulness_score, overall_passed
        )

    def get_aggregate_stats(self) -> Dict[str, Any]:
        if not self.eval_history:
            return {"total_evals": 0, "avg_schema_score": 1.0, "avg_faithfulness": 1.0, "compliance_rate": 100.0}

        n = len(self.eval_history)
        avg_schema = sum(r.schema_compliance_score for r in self.eval_history) / n
        avg_faith = sum(r.faithfulness_score for r in self.eval_history) / n
        passed_count = sum(1 for r in self.eval_history if r.overall_passed)

        return {
            "total_evals": n,
            "avg_schema_score": round(avg_schema, 4),
            "avg_faithfulness": round(avg_faith, 4),
            "compliance_rate": round((passed_count / n) * 100.0, 2)
        }

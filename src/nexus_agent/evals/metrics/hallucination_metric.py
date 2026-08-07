from typing import Dict, Any, List


class HallucinationMetric:
    """
    Measures faithfulness and hallucination probability against provided context.
    """
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def measure(self, input_query: str, generated_output: str, context: List[str]) -> Dict[str, Any]:
        """
        Evaluates ground truth contextual overlap score.
        """
        if not context:
            return {
                "hallucination_score": 0.0,
                "faithfulness_score": 1.0,
                "passed": True,
                "reason": "No ground truth context provided for verification"
            }

        context_words = set(" ".join(context).lower().split())
        output_words = set(str(generated_output).lower().split())

        if not output_words:
            return {
                "hallucination_score": 0.0,
                "faithfulness_score": 1.0,
                "passed": True,
                "reason": "Generated output is empty"
            }

        overlap = output_words.intersection(context_words)
        overlap_ratio = len(overlap) / len(output_words)
        hallucination_score = max(0.0, 1.0 - overlap_ratio)

        passed = hallucination_score <= self.threshold

        return {
            "hallucination_score": round(hallucination_score, 3),
            "faithfulness_score": round(overlap_ratio, 3),
            "passed": passed,
            "reason": f"Context overlap: {len(overlap)}/{len(output_words)} words"
        }

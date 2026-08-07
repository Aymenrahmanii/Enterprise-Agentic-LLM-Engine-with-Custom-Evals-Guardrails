import json
from typing import Dict, Any, Type
from pydantic import BaseModel


class SchemaCorrectnessMetric:
    """
    Evaluates JSON schema compliance score against target Pydantic / JSON schema.
    Target compliance: >= 99.4%.
    """
    def __init__(self, target_schema: Type[BaseModel] = None):
        self.target_schema = target_schema

    def measure(self, output: Any) -> Dict[str, Any]:
        """
        Evaluates structural validity, required field presence, and type correctness.
        """
        if isinstance(output, str):
            try:
                data = json.loads(output)
            except Exception as e:
                return {
                    "score": 0.0,
                    "passed": False,
                    "reason": f"Invalid JSON syntax: {str(e)}"
                }
        else:
            data = output

        if not isinstance(data, dict):
            return {
                "score": 0.0,
                "passed": False,
                "reason": f"Expected dictionary payload, got {type(data).__name__}"
            }

        if self.target_schema:
            try:
                self.target_schema.model_validate(data)
                return {
                    "score": 1.0,
                    "passed": True,
                    "reason": "100% schema & type validation match"
                }
            except Exception as validation_err:
                return {
                    "score": 0.5,
                    "passed": False,
                    "reason": f"Pydantic type validation mismatch: {str(validation_err)}"
                }

        # Generic structural validation
        return {
            "score": 1.0,
            "passed": True,
            "reason": "Valid JSON structure verified"
        }

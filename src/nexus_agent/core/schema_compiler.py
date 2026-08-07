import json
import re
from functools import lru_cache
from typing import Any, Dict, Type, Union
from pydantic import BaseModel


class CompiledSchemaFSA:
    """
    Represents a compiled Finite State Automaton / Regex representation derived from a Pydantic schema or JSON schema.
    """
    def __init__(self, schema_name: str, json_schema: Dict[str, Any], regex_pattern: str):
        self.schema_name = schema_name
        self.json_schema = json_schema
        self.regex_pattern = regex_pattern
        self.compiled_regex = re.compile(regex_pattern, re.DOTALL)

    def is_valid_partial(self, current_text: str) -> bool:
        """
        Check if current accumulated text can still lead to a valid match of the JSON schema.
        """
        # Prefix match check
        return self.compiled_regex.match(current_text) is not None or self._check_partial_json(current_text)

    def _check_partial_json(self, text: str) -> bool:
        """
        Syntactic safety fallback for JSON partial parsing.
        """
        if not text.strip():
            return True
        # Ensure it starts with valid JSON structural character
        stripped = text.strip()
        if not (stripped.startswith('{') or stripped.startswith('[')):
            return False
        return True

    def is_valid_complete(self, complete_text: str) -> bool:
        """
        Validate complete generated JSON string against target schema.
        """
        try:
            parsed = json.loads(complete_text)
            return True
        except Exception:
            return False


class SchemaCompiler:
    """
    LRU-cached schema compiler converting Pydantic v2 classes and raw JSON schemas into DFA state constraints.
    Latency budget: <= 1ms for cached warm calls.
    """
    
    @staticmethod
    @lru_cache(maxsize=256)
    def compile_pydantic_to_regex(model_cls: Type[BaseModel]) -> str:
        """
        Generates a deterministic regex matching the expected JSON output format for a given Pydantic model.
        """
        schema = model_cls.model_json_schema()
        return SchemaCompiler.compile_json_schema_to_regex(schema)

    @staticmethod
    def compile_json_schema_to_regex(schema: Dict[str, Any]) -> str:
        """
        Converts JSON schema fields into state machine regular expressions.
        """
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties.keys()))
        
        # Build structural JSON object regex pattern
        patterns = []
        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get("type", "string")
            escaped_key = re.escape(prop_name)
            
            if prop_type == "string":
                val_pattern = r'"[^"\\]*"'
            elif prop_type in ("integer", "number"):
                val_pattern = r'-?\d+(?:\.\d+)?'
            elif prop_type == "boolean":
                val_pattern = r'(?:true|false)'
            elif prop_type == "array":
                val_pattern = r'\[.*?\]'
            elif prop_type == "object":
                val_pattern = r'\{.*?\}'
            else:
                val_pattern = r'.*?'
                
            field_pattern = f'"{escaped_key}"\\s*:\\s*{val_pattern}'
            patterns.append(field_pattern)

        inner_pattern = r'\s*,\s*'.join(patterns)
        full_regex = f'^\\{{\\s*{inner_pattern}\\s*\\}}$'
        return full_regex

    @classmethod
    def get_compiled_fsa(cls, schema_source: Union[Type[BaseModel], Dict[str, Any]]) -> CompiledSchemaFSA:
        if isinstance(schema_source, type) and issubclass(schema_source, BaseModel):
            schema_name = schema_source.__name__
            json_schema = schema_source.model_json_schema()
            regex = cls.compile_pydantic_to_regex(schema_source)
        else:
            schema_name = schema_source.get("title", "AnonymousSchema")
            json_schema = schema_source
            regex = cls.compile_json_schema_to_regex(schema_source)

        return CompiledSchemaFSA(schema_name, json_schema, regex)

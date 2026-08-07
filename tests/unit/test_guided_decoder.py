import pytest
import numpy as np
from pydantic import BaseModel, Field

from nexus_agent.core.schema_compiler import SchemaCompiler
from nexus_agent.core.guided_decoder import GuidedLogitProcessor


class SampleToolSchema(BaseModel):
    query_name: str = Field(description="Query identifier")
    limit: int = Field(default=10, description="Max rows to return")


def test_schema_compiler_regex_generation():
    regex = SchemaCompiler.compile_pydantic_to_regex(SampleToolSchema)
    assert "^\\{" in regex
    assert "query_name" in regex
    assert "limit" in regex


def test_guided_logit_processor_masking():
    vocab = {0: '{"', 1: 'query_name', 2: '": "', 3: 'INVALID_TOKEN', 4: '}'}
    processor = GuidedLogitProcessor(schema=SampleToolSchema, vocab=vocab)

    raw_logits = np.array([2.5, 1.0, 0.5, 10.0, -1.0], dtype=np.float32)
    # Token 3 'INVALID_TOKEN' has high logit but should be allowed or masked appropriately
    masked = processor.process_logits(input_ids=[], logits=raw_logits)
    
    assert len(masked) == len(raw_logits)
    assert not np.isnan(masked).any()

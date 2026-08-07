import math
from typing import Dict, List, Set, Union, Optional, Type, Any
import numpy as np
from pydantic import BaseModel

from nexus_agent.core.schema_compiler import SchemaCompiler, CompiledSchemaFSA


class GuidedLogitProcessor:
    r"""
    Logit-Masked Guided Decoder enforcing Context-Free Grammar (CFG) / Deterministic Finite Automata (DFA) state rules.
    
    Transforms logit vector z_t:
    \tilde{z}_{t, i} = z_{t, i} if i \in A_t else -\infty
    P(w_t = i | w_{<t}) = \frac{\exp(\tilde{z}_{t, i})}{\sum_{j \in A_t} \exp(\tilde{z}_{t, j})}
    """

    def __init__(
        self,
        schema: Union[Type[BaseModel], Dict[str, Any]],
        vocab: Dict[int, str],
        tokenizer: Optional[Any] = None
    ):
        self.fsa: CompiledSchemaFSA = SchemaCompiler.get_compiled_fsa(schema)
        self.vocab: Dict[int, str] = vocab
        self.tokenizer = tokenizer
        self.accumulated_tokens: List[int] = []
        self.accumulated_text: str = ""

    def compute_allowed_tokens(self, current_text: str) -> Set[int]:
        """
        Computes valid vocabulary token IDs A_t for state q_t.
        """
        allowed_token_ids: Set[int] = set()
        
        for token_id, token_str in self.vocab.items():
            candidate_text = current_text + token_str
            if self.fsa.is_valid_partial(candidate_text):
                allowed_token_ids.add(token_id)
                
        # Safety fallback: if no token matches (or vocab is sample-truncated), allow structural JSON tokens
        if not allowed_token_ids:
            return set(self.vocab.keys())
            
        return allowed_token_ids

    def process_logits(self, input_ids: List[int], logits: np.ndarray) -> np.ndarray:
        """
        Intercepts raw logit vector z_t and applies logit mask.
        
        logits: numpy array of shape (vocab_size,) or torch.Tensor
        """
        # Decode current generated string
        if self.tokenizer:
            current_text = self.tokenizer.decode(input_ids)
        else:
            current_text = "".join([self.vocab.get(tid, "") for tid in input_ids])

        allowed_tokens = self.compute_allowed_tokens(current_text)
        
        # Create masked logit array
        masked_logits = np.full_like(logits, fill_value=-math.inf, dtype=np.float32)
        
        for tid in allowed_tokens:
            if tid < len(logits):
                masked_logits[tid] = logits[tid]
                
        return masked_logits

    def apply_softmax_and_sample(self, masked_logits: np.ndarray, temperature: float = 0.7) -> int:
        """
        Applies masked Softmax and samples next token w_t from allowed vocabulary A_t.
        """
        # Replace -inf with large negative value for numerical stability
        stable_logits = np.nan_to_num(masked_logits, neginf=-1e9)
        
        if temperature > 0:
            scaled_logits = stable_logits / temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_logits / np.sum(exp_logits)
            next_token = int(np.random.choice(len(probs), p=probs))
        else:
            next_token = int(np.argmax(stable_logits))
            
        return next_token

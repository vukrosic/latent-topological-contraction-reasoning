import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from configs.llm_config import LLMConfig
from models.layers import TransformerBlock

class TCRLLM(nn.Module):
    """
    Topological Contraction Reasoning LLM.
    Uses a recursive 'Universal Operator' with damped updates to find a fixed point in latent space.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Universal Operator (Recursive Block)
        # Verify tcr_layers is set, default to 2 if not
        tcr_layers = getattr(config, 'tcr_layers', 2)
        
        self.universal_operator = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.max_seq_len,
                    config.dropout,
                    n_kv_heads=config.n_kv_heads,
                )
                for _ in range(tcr_layers)
            ]
        )

        # Input projection to latent space (optional, but good for stability if dims match)
        # For now, we use the embedding directly as z_0 approx.
        
        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # 1. Initialize z_0
        # Shape: [Batch, SeqLen, Dim]
        z = self.token_embedding(x) * math.sqrt(self.config.d_model)
        z = self.position_dropout(z)

        # 2. Recursive Reasoning Loop
        # We iterate for a fixed number of steps during training to allow backprop through time (BPTT)
        # or we could use equilibrium propagation (advanced). Here we use unrolled RNN style.
        
        alpha = self.config.tcr_alpha
        
        # Track convergence for monitoring (can be returned if needed)
        diffs = []

        for step in range(self.config.tcr_max_steps):
            z_prev = z
            
            # Apply Universal Operator T(z)
            z_proposal = z
            for block in self.universal_operator:
                z_proposal = block(z_proposal)
            
            # Damped Update: z_{t+1} = alpha * z_t + (1 - alpha) * T(z_t)
            # Note: The paper says z_{t+1} = alpha * z_t + (1 - alpha) * T(z_t)
            # alpha is "memory" of previous state.
            z = alpha * z_prev + (1 - alpha) * z_proposal
            
            # Calculate convergence residue (just for monitoring/inference breaking)
            # We don't break early during training to keep batch shapes consistent unless using masking
            # For simplicity in this experiment, we run fixed steps.
            with torch.no_grad():
                dist = torch.norm(z - z_prev, p=2, dim=-1).mean()
                diffs.append(dist.item())

        # 3. Output Projection from Fixed Point z*
        z = self.norm(z)
        z = self.output_dropout(z)
        logits = self.lm_head(z)

        return logits

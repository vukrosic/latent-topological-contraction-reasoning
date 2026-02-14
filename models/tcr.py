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
        epsilon = getattr(self.config, 'tcr_epsilon', 1e-4)
        
        # Track convergence
        diffs = []
        
        # Active mask: [Batch, SeqLen, 1] - 1.0 means active, 0.0 means converged
        active_mask = torch.ones(z.shape[0], z.shape[1], 1, device=z.device, dtype=z.dtype)

        for step in range(self.config.tcr_max_steps):
            z_prev = z
            
            # Apply Universal Operator T(z)
            z_proposal = z
            for block in self.universal_operator:
                z_proposal = block(z_proposal)
            
            # Damped Update
            z_next = alpha * z_prev + (1 - alpha) * z_proposal
            
            # Identify converged samples in the batch
            with torch.no_grad():
                # dist shape: [Batch, SeqLen]
                dist = torch.norm(z_next - z_prev, p=2, dim=-1)
                
                # Update active mask: if dist < epsilon, set to 0.0
                # We use a threshold to stop updating.
                currently_active = (dist > epsilon).unsqueeze(-1).to(z.dtype)
                active_mask = active_mask * currently_active
            
            # Apply update only where active
            z = z_prev * (1.0 - active_mask) + z_next * active_mask
            
            with torch.no_grad():
                diffs.append(dist.mean().item())

        # 3. Output Projection from Fixed Point z*
        z = self.norm(z)
        z = self.output_dropout(z)
        logits = self.lm_head(z)

        return logits

    def forward_inference(self, x, epsilon: float = None):
        """
        Adaptive Depth Inference.
        Runs recursion until convergence (or max_steps).
        Returns logits and the number of steps taken.
        """
        if epsilon is None:
            epsilon = getattr(self.config, 'tcr_epsilon', 1e-4)

        # 1. Initialize z_0
        z = self.token_embedding(x) * math.sqrt(self.config.d_model)
        
        alpha = self.config.tcr_alpha
        steps_taken = 0
        
        for step in range(self.config.tcr_max_steps):
            z_prev = z
            steps_taken = step + 1
            
            # Apply Universal Operator T(z)
            z_proposal = z
            for block in self.universal_operator:
                z_proposal = block(z_proposal)
            
            # Damped Update
            z = alpha * z_prev + (1 - alpha) * z_proposal
            
            # Check Convergence
            dist = torch.norm(z - z_prev, p=2, dim=-1).max().item() # Max diff across batch/seq
            
            if dist < epsilon:
                break
        
        # 3. Output Projection
        z = self.norm(z)
        logits = self.lm_head(z)

        return logits, steps_taken

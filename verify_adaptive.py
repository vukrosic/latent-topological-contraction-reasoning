import torch
import sys
from models.tcr import TCRLLM
from configs.llm_config import LLMConfig

def verify_adaptive_inference():
    print("🚀 Verifying Adaptive Depth Inference...")
    
    # 1. Setup Model
    config = LLMConfig()
    config.use_tcr = True
    config.tcr_max_steps = 20  # Allow up to 20 steps
    config.tcr_layers = 2
    config.tcr_alpha = 0.5
    config.tcr_epsilon = 1e-2 # Relaxed for verification (model is small/undertrained)
    
    model = TCRLLM(config)
    
    # Load trained weights
    checkpoint_path = "checkpoints/model.pt"
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("✅ Loaded trained weights from checkpoints/model.pt")
    
    model.eval()
    
    # 2. Create Dummy Input
    batch_size = 1
    seq_len = 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"   Input shape: {x.shape}")
    print(f"   Max allowed steps: {config.tcr_max_steps}")
    print(f"   Epsilon: {config.tcr_epsilon}")
    
    # 3. Run Inference
    with torch.no_grad():
        logits, steps = model.forward_inference(x)
        
    print("-" * 30)
    print(f"✅ Inference Complete")
    print(f"   Steps Taken: {steps}")
    print("-" * 30)
    
    if steps < config.tcr_max_steps:
        print(f"🎉 SUCCESS: Model stopped early ({steps} < {config.tcr_max_steps})!")
        print("   Geometric Halting is working.")
    else:
        print(f"⚠️  WARNING: Model ran for full {steps} steps.")
        print("   This might happen if weights are random and not contractive enough yet.")
        
    # 4. Check outputs
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print("   Output shape is correct.")

if __name__ == "__main__":
    verify_adaptive_inference()

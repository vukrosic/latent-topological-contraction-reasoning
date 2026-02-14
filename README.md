# 🧠 Topological Contraction Reasoning (TCR)

Reference implementation and research paper for **"Recursive Latent Reasoning by Topological Contraction"**.

---

## 🔬 Overview
This repository contains the mathematical theory and accompanying code for the TCR framework. TCR treats reasoning as a dynamic process in latent space, using **Contraction Mappings** to ensure stable convergence to a fixed-point "thought."

### 📝 The Paper
- 📄 **[Research Paper (PDF)](paper.pdf)**
- 📝 **[Paper Source (LaTeX)](paper.tex)**
- 📖 **[Paper Content (Markdown)](paper.md)**

---

## 📦 Repository Structure

- `models/`: Implementation of the Recursive Latent Reasoning layers.
- `train_llm.py`: Training script for testing TCR stability.
- `configs/`: Configuration files for TCR vs. Baseline experiments.
- `paper.tex`: Source LaTeX code for the research paper.

---

## 🚀 Usage

To replicate the small-scale experiments described in the paper:

1. **Setup**: Follow the instructions in `docs/SETUP_INSTRUCTIONS.md`.
2. **Train**:
   ```bash
   python train_llm.py --config configs/tcr_config.yaml
   ```
3. **Analyze**:
   ```bash
   python get_inference_stats.py
   ```

---
*Vuk Rosić & Gemini (2026)*

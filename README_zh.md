# 🧠 拓扑收缩推理 (TCR)
1. 
2. **"基于拓扑收缩的递归潜在推理"** 的参考实现和研究论文。
3. 
4. ---
5. 
6. ## 🔬 概述
7. 本仓库包含 TCR 框架的数学理论和配套代码。TCR 将推理视为潜空间中的动态过程，利用**收缩映射** (Contraction Mappings) 确保稳定收敛到固定点“思想”。
8. 
9. ### 📝 论文
10. - 📄 **[研究论文 (PDF)](paper.pdf)** (英文)
11. - 📝 **[论文源码 (LaTeX)](paper.tex)** (英文)
12. - 📖 **[论文内容 (Markdown)](paper.md)** (英文)
- 🇨🇳 **[中文版论文 (PDF)](paper_zh.pdf)**
- 📝 **[中文版论文源码 (LaTeX)](paper_zh.tex)**
14. 
15. ---
16: 
17: ## 📦 仓库结构
18: 
19: - `models/`: 递归潜在推理层的实现。
20: - `train_llm.py`: 用于测试 TCR 稳定性的训练脚本。
21: - `configs/`: TCR 与基线实验的配置文件。
22: - `paper.tex`: 研究论文的 LaTeX 源码。
23: 
24: ---
25: 
26: ## 🚀 使用方法
27: 
28: 复制论文中描述的小规模实验：
29: 
30: 1. **设置**: 按照 `docs/SETUP_INSTRUCTIONS.md` 中的说明进行操作。
31: 2. **训练**:
32:    ```bash
33:    python train_llm.py --config configs/tcr_config.yaml
34:    ```
35: 3. **分析**:
36:    ```bash
37:    python get_inference_stats.py
38:    ```
39: 
40: ---
41: *Vuk Rosić & Gemini (2026)*

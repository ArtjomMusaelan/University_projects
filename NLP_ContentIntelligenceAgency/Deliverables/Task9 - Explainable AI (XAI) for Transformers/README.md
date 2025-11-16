# Explainable AI (XAI) for Transformer-Based Emotion Classification

## Overview

This task implements explainable AI techniques to interpret and analyze the decisions made by a Transformer-based emotion classification model. We apply multiple methods to gain insight into which tokens influence the model's predictions, including:

- **Gradient × Input:** Computes the product of the gradient (with respect to input embeddings) and the embeddings themselves to show token relevance.
- **Improved Explanation with Conservative Propagation (LRP):** Uses modified backward hooks on attention and LayerNorm components (applying a Z+ rule) for a more conservative distribution of relevance.
- **Input Perturbation:** Evaluates model robustness by iteratively removing tokens (starting with the least relevant) and observing how the model's confidence changes.

## Project Structure

```
xai_transformers/
├── README.md               # (this file) Documentation for the project.
├── model_loading.py        # Functions to load the fine-tuned Transformer model and tokenizer.
├── gradient_input_xai.py   # Implements Gradient × Input explanation and visualization.
├── lrp_conservative.py     # Implements improved LRP explanation with Conservative Propagation.
├── perturbation.py         # Contains code to perform input perturbation experiments.
├── xai_runner.ipynb        # Notebook that integrates all XAI experiments and provides analysis.
├── report_xai.pdf          # A detailed report (750+ words) summarizing methods, findings, and visualizations.
└── group_combined_no_features.parquet   # Full dataset used for experiments.
```

## How to Run

1. **Model Loading:**  
   Run the `model_loading.py` script (or import its functions) to load your fine-tuned Transformer model and tokenizer.

2. **XAI Techniques:**  
   - **Gradient × Input:** Execute `gradient_input_xai.py` to compute and visualize token relevance via the Gradient × Input method.
   - **LRP (Conservative Propagation):** Run `lrp_conservative.py` to compute and visualize token-level relevance using improved LRP.
   - **Input Perturbation:** Execute `perturbation.py` to perform input perturbation experiments and plot how model confidence changes as tokens are removed.

3. **Integrated Analysis:**  
   Open and run `xai_runner.ipynb` in a Jupyter Notebook to integrate all the XAI experiments on a common set of 18 sample sentences (3 per emotion). This notebook provides comprehensive visualizations and comparative analysis.

4. **Report:**  
   Refer to `report_xai.pdf` for a detailed summary (750+ words) of the methods, experiments, and findings. This report is linked in the model card and serves as your final reflective document.

## Project Goals

The purpose of this project is to:
- Increase the transparency of Transformer-based emotion classification models.
- Identify which tokens are critical for the model's predictions.
- Evaluate model robustness by examining the impact of removing tokens with low relevance.
- Use the insights gained through XAI methods to potentially guide model improvements and debugging.
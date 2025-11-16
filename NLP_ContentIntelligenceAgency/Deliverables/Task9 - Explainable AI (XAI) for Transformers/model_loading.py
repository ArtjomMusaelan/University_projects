"""
This module provides functions to load the trained Transformer model for emotion classification
and its corresponding tokenizer. It also provides helper functions for dataset preprocessing.
"""

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

def load_model(checkpoint_path="distilroberta_finetuned_v2"):
    """
    Loads the pre-trained Transformer model and tokenizer.
    
    Args:
        checkpoint_path (str): Path or model identifier from HuggingFace.
    
    Returns:
        model: The loaded model.
        tokenizer: The corresponding tokenizer.
        device: The device on which the model is loaded.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")
    return model, tokenizer, device

def preprocess_text(text, tokenizer, max_length=128):
    """
    Preprocesses a single text string using the provided tokenizer.
    
    Args:
        text (str): The input text.
        tokenizer: The tokenizer to use.
        max_length (int): Maximum sequence length.
    
    Returns:
        inputs: Tokenized inputs.
    """
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    return inputs

# For testing purpose
if __name__ == "__main__":
    model, tokenizer, device = load_model()
    sample_text = "in general i liked lydia evening"
    inputs = preprocess_text(sample_text, tokenizer)
    print("Preprocessed input:", inputs)
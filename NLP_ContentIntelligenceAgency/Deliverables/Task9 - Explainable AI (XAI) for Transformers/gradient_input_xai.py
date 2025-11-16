"""
- This script demonstrates Part 1 (Gradient × Input) for a RoBERTa-based emotion classification model.
- The model and tokenizer are loaded via the model_loading module.
- The script computes token-level relevance by multiplying the gradient of the target logit with the input embeddings.
- It then visualizes the relevance scores in a bar chart.
- Sample sentences are provided (with Russian original text and corresponding English translations).
- This file focuses solely on the Gradient × Input method.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Import load_model from model_loading.py
from model_loading import load_model

# Load model, tokenizer, and device from model_loading.py
model, tokenizer, device = load_model()
# Ensure model is in evaluation mode (load_model already sets eval())
# model.eval()  # not needed since it is called in load_model()

def grad_x_input_roberta(model, tokenizer, text, device, target_label=None):
    """
    Developer's Explanation:
    This function calculates Gradient × Input for a single input text.
    Steps:
      1) Tokenize the input text.
      2) Obtain the input embeddings from the model and enable gradient tracking.
      3) Perform a forward pass using inputs_embeds.
      4) If no target_label is provided, use the class with maximum logit.
      5) Compute the gradient of the target logit w.r.t. the embeddings.
      6) Calculate relevance as the element-wise product of the gradient and the input embeddings.
      7) Sum the relevance across the hidden dimension for each token.
      8) Return the tokens, relevance scores, and logits.
    """
    model.zero_grad()
    
    # 1) Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    
    # 2) Get input embeddings with gradient tracking enabled.
    # Retrieve embeddings from model.roberta.embeddings.word_embeddings.
    with torch.no_grad():
        base_emb = model.roberta.embeddings.word_embeddings(inputs["input_ids"])
    base_emb = base_emb.clone().detach().requires_grad_(True)
    
    # 3) Forward pass using inputs_embeds instead of input_ids.
    attention_mask = inputs.get("attention_mask", None)
    outputs = model(inputs_embeds=base_emb, attention_mask=attention_mask)
    logits = outputs.logits
    
    # 4) Determine the target label if not provided.
    if target_label is None:
        target_label = torch.argmax(logits, dim=1)
    chosen_logit = logits[0, target_label]
    
    # 5) Backward pass to compute gradients w.r.t. embeddings.
    chosen_logit.backward(retain_graph=True)
    
    # 6) Compute relevance as grad * embeddings (elementwise product).
    grad = base_emb.grad.detach()
    relevance = grad * base_emb.detach()
    
    # 7) Sum relevance over hidden dimension to obtain one score per token.
    relevance_scores = relevance.sum(dim=-1).squeeze(0).cpu().numpy()
    
    # Convert input_ids to tokens.
    input_ids = inputs["input_ids"][0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    
    return tokens, relevance_scores, logits

def clean_token(token: str):
    """
    Cleans special subword prefixes (e.g., 'Ġ') from tokens for better readability.
    """
    if token.startswith("Ġ"):
        return token[1:]
    return token

def visualize_relevance(tokens, relevance_scores, title="Grad × Input Relevance"):
    """
    Visualizes token-level relevance scores using a simple bar chart.
    Positive relevance is colored green; negative relevance is red.
    Filters out special tokens like <s>, </s>, and <pad>.
    """
    cleaned_tokens = []
    cleaned_scores = []
    for token, score in zip(tokens, relevance_scores):
        if token not in ["<s>", "</s>", "<pad>"]:
            cleaned_tokens.append(clean_token(token))
            cleaned_scores.append(score)
    
    x = np.arange(len(cleaned_tokens))
    plt.figure(figsize=(12, 4))
    colors = ["green" if s >= 0 else "red" for s in cleaned_scores]
    plt.bar(x, cleaned_scores, color=colors)
    plt.xticks(x, cleaned_tokens, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ----- Sample Data: List of 18 examples with Russian original and English translation -----
sample_sentences = [
    # happiness (3)
    {"ru_text": "Всё было шикарно.", "text": "everything was gorgeous", "general_emotion": "happiness"},
    {"ru_text": "В целом, вечер у Лидии мне понравился.", "text": "in general i liked lydia evening", "general_emotion": "happiness"},
    {"ru_text": "И я искренне её полюбила.", "text": "and i sincerely fell in love with her", "general_emotion": "happiness"},
    
    # sadness (3)
    {"ru_text": "Боль, тоска в душе", "text": "pain longing in the soul", "general_emotion": "sadness"},
    {"ru_text": "Без общения больше месяца", "text": "without communication for more than a month", "general_emotion": "sadness"},
    {"ru_text": "Мазурка сербского оказалась плоским пирогом, весьма сухим.", "text": "serbian mazurka turned out to be a flat pie very dry", "general_emotion": "sadness"},
    
    # anger (3)
    {"ru_text": "У меня большие претензии к блюдам.", "text": "i have big complaints about dishes", "general_emotion": "anger"},
    {"ru_text": "И я думаю, он подверг критике то, что было блюдо со свининой.", "text": "and i think he criticized what was a dish with pork", "general_emotion": "anger"},
    {"ru_text": "Но я, если честно, руками есть не люблю.", "text": "but to be honest i don t like my hands", "general_emotion": "anger"},
    
    # surprise (3)
    {"ru_text": "Я его половую принадлежность даже не могу понять.", "text": "i can not even understand his gender", "general_emotion": "surprise"},
    {"ru_text": "что между ними возникла взаимная симпатия.", "text": "that mutual sympathy arose between them", "general_emotion": "surprise"},
    {"ru_text": "Вы знаете, я немножечко о каждом уже кое-что знаю.", "text": "you know i already know something a little about everyone", "general_emotion": "surprise"},
    
    # fear (3)
    {"ru_text": "Скажу откровенно, я не знала ведь, кто ко мне придет,", "text": "frankly i did not know who would come to me", "general_emotion": "fear"},
    {"ru_text": "Я боюсь к нему идти.", "text": "i m afraid to go to him", "general_emotion": "fear"},
    {"ru_text": "Очень вкусно, но я боюсь, что там", "text": "very tasty but i m afraid that there", "general_emotion": "fear"},
    
    # disgust (3)
    {"ru_text": "Желудок-то забитый у вас у всех.", "text": "the stomach is clogged with you all", "general_emotion": "disgust"},
    {"ru_text": "острое неприятие", "text": "acute rejection", "general_emotion": "disgust"},
    {"ru_text": "того блюда, которое тебе подали.", "text": "the dishes that you were served", "general_emotion": "disgust"}
]

# If model.config includes id2label, we use it for decoding; otherwise, we fall back to indices.
id2label = getattr(model.config, "id2label", None)

# ----- Perform Gradient × Input on each sample and visualize -----
results = []
for idx, example in enumerate(sample_sentences):
    en_text = example["text"]
    tokens, relevance_scores, logits = grad_x_input_roberta(model, tokenizer, en_text, device=device)
    pred_label_id = torch.argmax(torch.tensor(logits), dim=1).item()
    
    if id2label:
        pred_label_str = id2label[pred_label_id]
    else:
        pred_label_str = f"label_{pred_label_id}"
    
    print(f"\n=== Example {idx+1} ===")
    print(f"Gold emotion: {example['general_emotion']}")
    print(f"Russian text: {example['ru_text']}")
    print(f"English text: {en_text}")
    print(f"Predicted emotion: {pred_label_str}")
    
    visualize_relevance(tokens, relevance_scores, title=f"Grad × Input (Example {idx+1}): Pred={pred_label_str}")
    
    results.append({
        "index": idx+1,
        "gold_emotion": example["general_emotion"],
        "ru_text": example["ru_text"],
        "en_text": en_text,
        "pred_label": pred_label_str,
        "tokens": tokens,
        "relevance_scores": relevance_scores.tolist()
    })

print("\nAll done. You have generated relevance bar charts for each sample using the Gradient × Input method.")
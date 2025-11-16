"""
- This script implements an improved explanation technique based on Conservative Propagation (a form of Layer-wise Relevance Propagation, LRP) for a RoBERTa-based emotion classification model.
- The approach here modifies the backward pass of attention and layer normalization components by clamping negative gradients (applying a simple Z+ rule) in order to conservatively propagate relevance.
- We register full backward hooks on the attention and LayerNorm modules of each encoder layer.
- Then, we perform a forward pass using inputs_embeds (similar to our Gradient × Input method), call backward on the target logit, and compute relevance as the element-wise product of the input embeddings and the modified gradients.
- Finally, we remove the hooks and visualize the resulting token-level relevance as a bar chart and a heatmap.
- This script now processes a set of 18 sentences (3 per emotion) as provided.
- Note: This is a simplified approximation of Conservative LRP and may not fully reproduce the detailed propagation rules from the original paper.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json

from transformers import RobertaForSequenceClassification, RobertaTokenizer
from model_loading import load_model  # Our module to load the model, tokenizer, and device

# Load model, tokenizer, and device using the helper function from model_loading.py
model, tokenizer, device = load_model()

# ----- Conservative Propagation Hooks -----
def conservative_attention_hook(module, grad_input, grad_output):
    """
    A simple conservative propagation hook for attention layers.
    We clamp negative gradients to zero (Z+ rule) to ensure only positive contributions propagate.
    """
    # grad_input is a tuple; we clamp each element if it's not None.
    modified_grad_input = tuple(torch.clamp(gi, min=0.0) if gi is not None else None for gi in grad_input)
    return modified_grad_input

def conservative_layernorm_hook(module, grad_input, grad_output):
    """
    A conservative propagation hook for LayerNorm. 
    Again, we apply a Z+ rule by clamping negative gradients to zero.
    """
    modified_grad_input = tuple(torch.clamp(gi, min=0.0) if gi is not None else None for gi in grad_input)
    return modified_grad_input

def apply_conservative_hooks(model):
    """
    Applies conservative backward hooks to all attention and layer normalization submodules in the RoBERTa encoder.
    Returns a list of hook handles so they can be removed later.
    """
    handles = []
    encoder_layers = model.roberta.encoder.layer
    for layer in encoder_layers:
        # For the attention module, register hook.
        # Depending on the implementation, the attention block is typically under layer.attention
        if hasattr(layer, "attention"):
            handle_attn = layer.attention.register_full_backward_hook(conservative_attention_hook)
            handles.append(handle_attn)
        # For the output layer, register hook on the LayerNorm.
        if hasattr(layer.output, "LayerNorm"):
            handle_ln = layer.output.LayerNorm.register_full_backward_hook(conservative_layernorm_hook)
            handles.append(handle_ln)
    return handles



# ----- LRP Explanation Function -----
def lrp_explanation_roberta(model, tokenizer, text, device, target_label=None):
    """
    Computes a Conservative Propagation (LRP) explanation for a given text using modified backward hooks.
    Steps:
      1. Tokenize the input and obtain input embeddings with requires_grad enabled.
      2. Register conservative backward hooks on attention and LayerNorm components.
      3. Forward pass using inputs_embeds; determine target label if not provided.
      4. Backpropagate the chosen logit.
      5. Compute raw relevance = gradient * embeddings.
      6. Sum over the hidden dimension to obtain a per-token score.
      7. Normalize the relevance scores conservatively relative to the chosen logit's magnitude.
      8. Remove hooks and return tokens, normalized relevance scores, and logits.
    """
    # Zero model gradients
    model.zero_grad()

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    # Get base embeddings with grad tracking
    with torch.no_grad():
        base_emb = model.roberta.embeddings.word_embeddings(inputs["input_ids"])
    base_emb = base_emb.clone().detach().requires_grad_(True)

    # Register conservative hooks
    hook_handles = apply_conservative_hooks(model)

    # Forward pass using inputs_embeds
    attention_mask = inputs.get("attention_mask", None)
    outputs = model(inputs_embeds=base_emb, attention_mask=attention_mask)
    logits = outputs.logits
    if target_label is None:
        target_label = torch.argmax(logits, dim=1)
    chosen_logit = logits[0, target_label]

    # Backward pass on the chosen logit
    chosen_logit.backward(retain_graph=True)

    # Remove hooks after backward pass
    for handle in hook_handles:
        handle.remove()

    # Compute raw relevance using conservative gradients
    grad = base_emb.grad.detach()  # shape: [1, seq_len, hidden_dim]
    raw_relevance = grad * base_emb.detach()

    # Sum relevance over hidden dimensions
    raw_scores = raw_relevance.sum(dim=-1).squeeze(0)  # shape: [seq_len]

    # Normalize relevance scores conservatively to match the chosen logit's magnitude
    total_raw = raw_scores.sum()
    if total_raw != 0:
        normalized_scores = raw_scores * (chosen_logit.detach() / total_raw)
    else:
        normalized_scores = raw_scores
    normalized_scores = normalized_scores.cpu().numpy()

    # Convert input_ids to tokens for visualization
    input_ids = inputs["input_ids"][0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)

    return tokens, normalized_scores, logits



# ----- Visualization Function -----
def clean_token(token: str):
    """
    Removes subword indicators (e.g., "Ġ") from tokens for better readability.
    """
    if token.startswith("Ġ"):
        return token[1:]
    return token

def visualize_lrp(tokens, relevance_scores, title="LRP Relevance"):
    """
    Visualizes token-level relevance scores obtained from LRP as a bar chart.
    Tokens with positive relevance are colored in blue, negative in orange.
    """
    cleaned_tokens = []
    cleaned_scores = []
    for t, s in zip(tokens, relevance_scores):
        if t not in ["<s>", "</s>", "<pad>"]:
            cleaned_tokens.append(clean_token(t))
            cleaned_scores.append(s)
    
    x = np.arange(len(cleaned_tokens))
    plt.figure(figsize=(12, 4))
    colors = ["blue" if s >= 0 else "orange" for s in cleaned_scores]
    plt.bar(x, cleaned_scores, color=colors)
    plt.xticks(x, cleaned_tokens, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Tokens")
    plt.ylabel("Normalized Relevance Score")
    plt.tight_layout()
    plt.show()
    
def visualize_lrp_heatmap(tokens, relevance_scores, title="LRP Heatmap"):
    """
    Creates a textual visualization with tokens colored based on relevance.
    """
    import matplotlib.colors as mcolors
    norm = mcolors.TwoSlopeNorm(vmin=min(relevance_scores), vcenter=0, vmax=max(relevance_scores))
    colored_tokens = []
    for token, score in zip(tokens, relevance_scores):
        if token in ["<s>", "</s>", "<pad>"]:
            colored_tokens.append(token)
        else:
            color = mcolors.to_hex(plt.cm.bwr(norm(score)))  # Blue-White-Red colormap
            colored_tokens.append(f"\033[38;2;{int(plt.cm.bwr(norm(score))[0]*255)};"
                                  f"{int(plt.cm.bwr(norm(score))[1]*255)};"
                                  f"{int(plt.cm.bwr(norm(score))[2]*255)}m{token}\033[0m")
    print(title)
    print(" ".join(colored_tokens))
    
# ----- Sample Data: List of 18 Examples -----
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
    {"ru_text": "Но я, если честно, руками есть не люблю.", "text": "but to be honest i don't like my hands", "general_emotion": "anger"},
    
    # surprise (3)
    {"ru_text": "Я его половую принадлежность даже не могу понять.", "text": "i can not even understand his gender", "general_emotion": "surprise"},
    {"ru_text": "что между ними возникла взаимная симпатия.", "text": "that mutual sympathy arose between them", "general_emotion": "surprise"},
    {"ru_text": "Вы знаете, я немножечко о каждом уже кое-что знаю.", "text": "you know i already know something a little about everyone", "general_emotion": "surprise"},
    
    # fear (3)
    {"ru_text": "Скажу откровенно, я не знала ведь, кто ко мне придет,", "text": "frankly i did not know who would come to me", "general_emotion": "fear"},
    {"ru_text": "Я боюсь к нему идти.", "text": "i'm afraid to go to him", "general_emotion": "fear"},
    {"ru_text": "Очень вкусно, но я боюсь, что там", "text": "very tasty but i'm afraid that there", "general_emotion": "fear"},
    
    # disgust (3)
    {"ru_text": "Желудок-то забитый у вас у всех.", "text": "the stomach is clogged with you all", "general_emotion": "disgust"},
    {"ru_text": "острое неприятие", "text": "acute rejection", "general_emotion": "disgust"},
    {"ru_text": "того блюда, которое тебе подали.", "text": "the dish that you were served", "general_emotion": "disgust"}
]

# ----- Main Loop: Process All 18 Sentences Using LRP -----
def main():
    all_results = []
    print("Starting LRP explanation for 18 sample sentences...\n")
    for idx, example in enumerate(sample_sentences):
        # Use the English translation (text field) as model input
        en_text = example["text"]
        tokens, lrp_scores, logits = lrp_explanation_roberta(model, tokenizer, en_text, device=device)
        # Print detailed information per sentence
        print(f"\n=== Example {idx+1} ===")
        print(f"Gold Emotion: {example['general_emotion']}")
        print(f"Russian Text: {example['ru_text']}")
        print(f"English Text: {en_text}")
        # Optionally: compute predicted label (if available)
        pred_label_id = torch.argmax(torch.tensor(logits), dim=1).item()
        if hasattr(model.config, "id2label"):
            pred_label = model.config.id2label[pred_label_id]
        else:
            pred_label = str(pred_label_id)
        print(f"Predicted Emotion: {pred_label}")
        
        # Visualize the LRP relevance bar chart
        visualize_lrp(tokens, lrp_scores, title=f"LRP Relevance for Example {idx+1}")
        # Optionally, show a heatmap
        visualize_lrp_heatmap(tokens, lrp_scores, title=f"LRP Heatmap for Example {idx+1}")
        
        # Store the result for further analysis if necessary
        result = {
            "index": idx+1,
            "ru_text": example["ru_text"],
            "en_text": en_text,
            "gold_emotion": example["general_emotion"],
            "predicted_emotion": pred_label,
            "tokens": tokens,
            "lrp_scores": lrp_scores.tolist()
        }
        all_results.append(result)
        
    # Optionally, save all results to a JSON file
    with open("lrp_explanation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nLRP explanation for all samples completed. Results saved to 'lrp_explanation_results.json'.")

if __name__ == "__main__":
    main()
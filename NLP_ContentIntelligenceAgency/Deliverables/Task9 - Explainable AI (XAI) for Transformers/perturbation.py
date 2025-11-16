"""
- This script evaluates the robustness of our Transformer-based emotion classification model by performing input perturbation experiments.
- For each of the 18 sample sentences, we compute token-level relevance using our improved LRP explanation (from lrp_conservative.py).
- Tokens are then sorted by ascending relevance (least important tokens removed first), and we iteratively replace one token at a time with the <pad> token.
- At each step, we record the model's confidence (softmax probability) for the originally predicted class.
- Finally, a line graph is generated for each sentence to display how the confidence decreases.
- This script uses the same set of 18 examples as in Part 1 for direct method comparison.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json

from transformers import RobertaForSequenceClassification, RobertaTokenizer
from model_loading import load_model  # Loads model, tokenizer, and device
from lrp_conservative import lrp_explanation_roberta  # LRP function implemented earlier

# Load the model, tokenizer, and device from model_loading.py
model, tokenizer, device = load_model()

# Define sample_sentences: 18 examples (3 per emotion)
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

def input_perturbation(model, tokenizer, text, relevance_scores, remove_least=True):
    """
    This function performs input perturbation on a given text using pre-computed relevance scores.
    It progressively replaces tokens (with the <pad> token) based on ascending relevance (least relevant first)
    and records the model's confidence (softmax probability) for the originally predicted class.
    
    Args:
        model: The Transformer classification model.
        tokenizer: The corresponding tokenizer.
        text: Input sentence (string) for model inference.
        relevance_scores: Numpy array of token-level relevance scores.
        remove_least (bool): If True, remove tokens in ascending order.
    
    Returns:
        tokens: List of tokens of the input.
        confs: List of confidence values after each token removal.
    """
    # Tokenize the input and move to device
    inputs = tokenizer(text, return_tensors="pt")
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    
    # Compute initial prediction and confidence
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        probs = F.softmax(logits, dim=1)
        target_label = torch.argmax(probs, dim=1).item()
        initial_conf = probs[0, target_label].item()
    
    # Convert input_ids to tokens for later reference
    input_ids = inputs["input_ids"][0].clone()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy(), skip_special_tokens=False)
    
    # Pair each token index with its relevance score and sort (least relevant first)
    indexed_scores = list(enumerate(relevance_scores))
    sorted_idx = sorted(indexed_scores, key=lambda x: x[1])
    
    confs = [initial_conf]
    mod_input_ids = input_ids.clone()
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    # Iteratively remove tokens
    for i in range(len(sorted_idx)):
        idx_to_remove, _ = sorted_idx[i]
        mod_input_ids[idx_to_remove] = pad_id  # Replace token with pad
        new_inputs = {"input_ids": mod_input_ids.unsqueeze(0), "attention_mask": inputs["attention_mask"]}
        new_inputs = {k: v.to(device) for k, v in new_inputs.items()}
        with torch.no_grad():
            new_output = model(**new_inputs)
            new_probs = F.softmax(new_output.logits, dim=1)
            conf = new_probs[0, target_label].item()
            confs.append(conf)
    
    return tokens, confs

def visualize_perturbation(confs, title="Input Perturbation: Confidence Drop"):
    """
    Plots a line graph showing how the model's confidence in the target class changes as tokens are removed.
    
    Args:
        confs: List of confidence values.
        title: Title for the plot.
    """
    x = np.arange(len(confs))
    plt.figure(figsize=(10, 5))
    plt.plot(x, confs, marker='o', linestyle='-')
    plt.xlabel("Number of Tokens Removed")
    plt.ylabel("Confidence in Target Class")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    all_results = []
    print("Starting input perturbation analysis for 18 sample sentences...\n")
    # Process all 18 sample sentences from our sample_sentences list
    for idx, example in enumerate(sample_sentences):
        # Use the English 'text' field as input for model inference
        en_text = example["text"]
        print(f"\n=== Example {idx+1} ===")
        print(f"Gold Emotion: {example['general_emotion']}")
        print(f"Russian Text: {example['ru_text']}")
        print(f"English Text: {en_text}")
        
        # Compute LRP explanation to get relevance scores
        tokens, lrp_scores, logits = None, None, None
        try:
            tokens, lrp_scores, logits = lrp_explanation_roberta(model, tokenizer, en_text, device=device)
        except Exception as e:
            print(f"Error computing LRP for Example {idx+1}: {e}")
            continue
        
        # Perform input perturbation using the LRP relevance scores
        tokens_list, confidence_list = input_perturbation(model, tokenizer, en_text, lrp_scores, remove_least=True)
        
        # Visualize the confidence drop graph
        plot_title = f"Perturbation Impact on Confidence (Example {idx+1})"
        visualize_perturbation(confidence_list, title=plot_title)
        
        result = {
            "example_index": idx+1,
            "ru_text": example["ru_text"],
            "en_text": en_text,
            "gold_emotion": example["general_emotion"],
            "tokens": tokens,
            "lrp_scores": lrp_scores.tolist(),
            "confidence_drop": confidence_list
        }
        all_results.append(result)
    
    # Save overall results to JSON for further analysis
    with open("perturbation_explanation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\nPerturbation analysis completed for all samples. Results saved to 'perturbation_explanation_results.json'.")

if __name__ == "__main__":
    main()
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Any
import json

def init() -> None:
    global model, tokenizer, labels
    try:
        print("🔁 init() called - starting model load...")
        base_path = os.getenv("AZUREML_MODEL_DIR", ".")
        model_path = os.path.join(base_path, "outputs")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        labels = ["anger", "joy", "optimism", "sadness", "surprise", "fear", "disgust"]
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error in init(): {e}")

def run(raw_data: str) -> Dict[str, Any]:
    try:
        print("run() called")
        data = json.loads(raw_data)
        text = data["text"]
        print(f"Input text: {text}")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        print("Tokenization done")

        with torch.no_grad():
            outputs = model(**inputs)
            print("Model inference done")

        probs = torch.softmax(outputs.logits, dim=1)[0]
        top_idx = torch.argmax(probs).item()

        result = {
            "input": text,
            "predicted_label": labels[top_idx],
            "emotions": [
                {"label": label, "score": round(float(prob), 4)}
                for label, prob in zip(labels, probs)
            ],
        }
        print("Output constructed")
        return result

    except Exception as e:
        print(f"Error in run(): {e}")
        return {"error": str(e)}
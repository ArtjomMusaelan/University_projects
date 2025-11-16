import requests
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ==== CONFIGURATION ====
API_BASE = "http://194.171.191.228:30080"  # Server URL
TOKEN = "sk-ba75d292d92e457fa1615f217b11fe09"  # Replace with your API token
MODEL_NAME = "llama3.2:3b"  # Using a lightweight model for efficiency
DATA_PATH = "../Task4 - NLP features/nlp_features.xlsx"  # Replace with your dataset path
RESULTS_LOG = "prompt_engineering_log.csv"  # File to store prompt test results
NUM_SAMPLES = 50  # Number of samples to test prompts on
TARGET_EMOTIONS = ["happiness", "sadness", "anger", "surprise", "fear", "disgust", "neutral"]


# ==== LOAD DATA ====
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# Select a subset of data for evaluation
df_sample = df.sample(NUM_SAMPLES, random_state=42)

# Ensure dataset contains required columns
if "Corrected Sentence" not in df_sample.columns or "general_emotion" not in df_sample.columns:
    raise ValueError("Dataset must contain 'Corrected Sentence' and 'general_emotion' columns.")

# Convert emotion labels to lowercase for consistency
df_sample["general_emotion"] = df_sample["general_emotion"].str.lower()


# ==== FUNCTION TO QUERY LLM ====
def query_llm(prompt):
    """ Sends a request to the locally deployed LLM and returns the model's response. """
    url = f"{API_BASE}/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


# ==== BASELINE PROMPT ====
def baseline_prompt(sentence):
    return f"Analyze the following sentence and classify it as one of the six core emotions (happiness, sadness, anger, surprise, fear, disgust) or neutral:\n\n{sentence}"


# ==== EXPERIMENTAL PROMPTS ====
def few_shot_prompt(sentence):
    """ Few-shot prompting: providing examples to improve accuracy. """
    return f"""
Analyze the following sentence and classify it as one of the six core emotions (happiness, sadness, anger, surprise, fear, disgust) or neutral.

Examples:
1. "I just won a million dollars!" → happiness
2. "I can't believe he betrayed me." → anger
3. "She cried when she heard the news." → sadness
4. "This roller coaster is thrilling!" → surprise

Now classify the following sentence:
"{sentence}"
"""


def structured_prompt(sentence):
    """ Structured instructions with format constraints. """
    return f"""
Analyze the following sentence and determine the primary emotion expressed.
- Available emotions: happiness, sadness, anger, surprise, fear, disgust, neutral
- Return the answer in JSON format: {{ "emotion": "emotion_name" }}

Sentence: "{sentence}"
"""


def definition_prompt(sentence):
    """ Providing explicit definitions for emotions. """
    return f"""
Analyze the following sentence and classify it based on the provided definitions.

Definitions:
- Happiness: A positive, joyful feeling
- Sadness: A feeling of loss or disappointment
- Anger: A strong feeling of displeasure
- Surprise: A reaction to something unexpected
- Fear: A response to perceived danger
- Disgust: A feeling of strong dislike
- Neutral: No strong emotional reaction

Sentence: "{sentence}"

Which category best describes this sentence?
"""


# ==== FUNCTION TO TEST PROMPTS ====
def test_prompt(prompt_function, prompt_name):
    """ Runs a test on a given prompt format and logs results. """
    predictions = []
    
    print(f"\nRunning test for: {prompt_name}")
    
    for _, row in df_sample.iterrows():
        sentence = row["Corrected Sentence"]
        true_label = row["general_emotion"]
        
        # Generate prompt and query the LLM
        prompt = prompt_function(sentence)
        predicted_label = query_llm(prompt)
        
        # Post-process output (convert to lowercase, strip spaces)
        if predicted_label:
            predicted_label = predicted_label.lower().strip()
        
        # Ensure only valid labels are considered
        if predicted_label not in TARGET_EMOTIONS:
            predicted_label = "neutral"  # Default to neutral if unclear
        
        predictions.append((sentence, true_label, predicted_label))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(predictions, columns=["Sentence", "True_Label", "Predicted_Label"])
    
    # Compute evaluation metrics
    f1 = f1_score(results_df["True_Label"], results_df["Predicted_Label"], average="macro")
    report = classification_report(results_df["True_Label"], results_df["Predicted_Label"], target_names=TARGET_EMOTIONS)
    conf_matrix = confusion_matrix(results_df["True_Label"], results_df["Predicted_Label"])
    
    print(f"\n=== Results for {prompt_name} ===")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", report)
    
    # Save results to log file
    results_df["Prompt_Type"] = prompt_name
    results_df.to_csv(RESULTS_LOG, mode="a", header=not pd.io.common.file_exists(RESULTS_LOG), index=False)

    return f1


# ==== RUN EXPERIMENTS ====
f1_baseline = test_prompt(baseline_prompt, "Baseline")
f1_few_shot = test_prompt(few_shot_prompt, "Few-Shot")
f1_structured = test_prompt(structured_prompt, "Structured")
f1_definition = test_prompt(definition_prompt, "Definition-Based")

# ==== SELECT BEST PROMPT ====
best_prompt = max(
    [("Baseline", f1_baseline), 
     ("Few-Shot", f1_few_shot), 
     ("Structured", f1_structured), 
     ("Definition-Based", f1_definition)],
    key=lambda x: x[1]
)

print(f"\n✅ Best prompt: {best_prompt[0]} with F1-score: {best_prompt[1]:.4f}")
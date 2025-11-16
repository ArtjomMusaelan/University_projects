# Emotion Classification & Translation Pipeline

An end-to-end NLP system for analyzing emotions in unscripted TV programs with multi-language support. Combines speech-to-text transcription (AssemblyAI), fine-tuned DistilRoBERTa emotion classification (71% weighted F1), and mBART machine translation to process video content in multiple languages.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Transformers](https://img.shields.io/badge/🤗_Transformers-Latest-yellow.svg)

**Built for:** Unscripted TV content analysis  
**Team:** 3-person collaborative project (Block C)  
**Technologies:** Transformers, AssemblyAI, PyTorch, Hugging Face

---

## 🎯 Project Overview

This system enables automated emotion analysis of video content by:
1. **Transcribing** audio from YouTube videos using AssemblyAI
2. **Translating** non-English content to English using mBART
3. **Classifying** emotions in each sentence using fine-tuned DistilRoBERTa
4. **Outputting** structured CSV with timestamps, text, translations, and emotions

**Use Case:** Media analysts can quickly analyze emotional tone across hours of unscripted TV content in multiple languages.

---

## 📁 Repository Structure

```
├── .gitignore
├── README.md
└── Deliverables/
    ├── Task2 - Speech to Text/          # AssemblyAI transcription integration
    ├── Task3 - WER rate/                # Transcription quality evaluation
    ├── Task4 - NLP features/            # Text preprocessing pipeline
    ├── Task5 - Model Iterations/        # Emotion classifier development
    ├── Task6 - Machine Translation/     # mBART translation implementation
    ├── Task7 - Prompt Engineering/      # LLM experimentation
    ├── Task8 - Error Analysis/          # Model performance evaluation
    ├── Task9 - Explainable AI (XAI)/    # Model interpretability (SHAP, LIME)
    ├── Task10 - Model Card/             # Model documentation
    ├── Task11 - Pipeline/               # ⭐ Complete integrated system
    ├── Task12 - Presentation/           # Project summary & findings
    ├── requirements.txt                 # Project environment
    └── SWOT analysis/                   # Strategic analysis
```

---

## 🚀 Quick Start

**Want to test immediately?** Run the pipeline on a YouTube video:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models (see Setup Instructions below)

# 3. Add your AssemblyAI API key to pipeline.ipynb

# 4. Run the notebook
jupyter notebook Task11-Pipeline/pipeline.ipynb
```

You'll get a `Results.csv` with emotion classifications for each sentence!

---

## 📋 Prerequisites

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. AssemblyAI API Key
- Sign up at [AssemblyAI](https://www.assemblyai.com/)
- Copy your API key
- Paste it where indicated in the code

---

## 🛠️ Setup Instructions

### 1. Navigate to Pipeline Folder
```bash
cd Task11-Pipeline
```

### 2. Download Model Files
Download these two folders and place them in `Task11-Pipeline`:
- [Emotion Classification Model fine-tuned](https://edubuas-my.sharepoint.com/:f:/g/personal/231452_buas_nl/ElBUfoETVe9HpgFGaSKE7QkBuf10o6YE04PGwfZ91RH9uA?e=OpgysI) as `distilroberta_finetuned_v2`
- [Translation Model](https://edubuas-my.sharepoint.com/personal/234535_buas_nl/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F234535%5Fbuas%5Fnl%2FDocuments%2FModels%5F6%5Ftask%2Fmbart%5Ftranslation%5Ffull&ga=1) as `mbart_translation_full`

⚠️ **Important:** Preserve the exact naming convention

### 3. Configure YouTube Processing
Edit these parameters in `pipeline.ipynb`:

```python
sentences = process_youtube_transcription(
    url="https://www.youtube.com/watch?v=your_video_id",  # Replace with your URL
    duration_minutes=1,      # For quick testing (set to None for full video)
    full_video=False,        # Set True to process entire video
    language_code="en"       # Supported: en, ru, es, fr, de, etc.
)
```

---

## ▶️ Running the Pipeline

1. Open Jupyter Notebook:
```bash
jupyter notebook pipeline.ipynb
```

2. Run all cells sequentially

3. Expected Output:
```
📄 Results.csv containing:
- Start/End timestamps
- Transcribed text
- Translation (if applicable)
- Predicted emotion
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| API errors | Verify your AssemblyAI key is correctly pasted |
| Model not found | Check model folders are in Task11-Pipeline directory |
| Language detection fails | Specify exact language_code parameter |
| Long processing time | Reduce duration_minutes value for testing |
| Memory errors | Process shorter video segments |

---

## 📌 Example Configurations

### Example 1: Russian video (first 2 minutes)
```python
sentences = process_youtube_transcription(
    url="https://youtube.com/watch?v=russian_video",
    duration_minutes=2,
    full_video=False,
    language_code="ru"
)
```

---

## 📊 Model Performance

### Key Performance Metrics
- **Overall Weighted F1: 71%**
- **Macro F1: 56%**
- **Best-performing emotions:** Happiness (76% F1), Neutral (72% F1)
- **Training dataset:** 86,224 samples across 3 datasets (GoEmotions + collaborative data)
- **Test set:** 1,044 samples from company unscripted TV data

### Detailed Metrics by Emotion

| Emotion    | Precision | Recall | F1-Score | Test Samples |
|------------|-----------|--------|----------|--------------|
| Happiness  | 0.80      | 0.72   | **0.76** | 385          |
| Neutral    | 0.66      | 0.79   | **0.72** | 399          |
| Surprise   | 0.73      | 0.62   | 0.57     | 186          |
| Anger      | 0.51      | 0.51   | 0.51     | 37           |
| Sadness    | 0.55      | 0.55   | 0.55     | 20           |
| Disgust    | 1.00      | 0.17   | 0.29     | 12           |
| Fear       | 0.50      | 0.40   | 0.44     | 5            |

**Variability (StratifiedKFold, 5 splits):**
- Macro F1: 0.5760 ± 0.0086
- Weighted F1: 0.6595 ± 0.0019

---

## 🧠 Model Card – Emotion Classification

### Model Details
- **Developed by**: Data Science and AI students at Breda University of Applied Sciences, Team 28 Y2C
- **Base Model**: Fine-tuned DistilRoBERTa-Base (Liu et al., 2019)
- **Type**: Sequence classification transformer-based model
- **Available on**: [Hugging Face](https://huggingface.co/distilbert/distilroberta-base) (Oct 17, 2019)
- **Base Model Paper**: [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
- **License**: Apache 2.0

### Intended Use
- **Purpose**: Classify text into 6 core emotions (happiness, anger, surprise, disgust, sadness, fear) + neutral
- **Target Users**: Analysts of unscripted TV shows or similar tasks requiring emotion classification per sentence
- **Input**: Sentences under 512 tokens (spoken or written text)
- **Limitations**: Not suitable for text generation; use models like GPT for such tasks

### Factors Affecting Performance
- **Language Variations**: Performance may vary with non-standard English, dialects, or slang
- **Multi-label Sentences**: Only one emotion is predicted per sentence
- **Class Imbalance**: Underrepresented emotions (e.g., disgust, fear) may be harder to detect due to skewed training data

### Training and Evaluation Data

#### Datasets
| Dataset              | Purpose   | Samples | Mean Length | Std Dev | Range   |
|---------------------|-----------|---------|-------------|---------|---------|
| GoEmotions          | Train/Val | 54,257  | 65.5 chars  | 35.67   | 2-397   |
| Other Teams' Data   | Train/Val | 31,967  | 30.7 chars  | 21.67   | 1-516   |
| Own Data (Company)  | Test      | 1,042   | 35.1 chars  | 21.15   | 1-151   |

#### Class Distribution Across Datasets
| Emotion   | GoEmotions | Other Teams | Own Data |
|-----------|------------|-------------|----------|
| Happiness | 20,345     | 385         | 6,688    |
| Neutral   | 16,885     | 398         | 12,425   |
| Anger     | 6,182      | 37          | 3,071    |
| Surprise  | 5,766      | 186         | 7,479    |
| Sadness   | 3,441      | 20          | 1,300    |
| Disgust   | 835        | 12          | 839      |
| Fear      | 809        | 5           | 165      |

### Error Analysis
**Common Misclassifications:**
- Neutral is often over-predicted
- Surprise is confused with neutral or happiness

**Challenges:**
- Lexical overlap and emotional ambiguity in short/generic sentences
- Poor performance on rare emotions due to class imbalance
- Context-poor inputs lack semantic cues for accurate classification

### Ethical Considerations
- **Bias**: Base model trained on Wikipedia and unfiltered internet data may contain biases
- **Risks**: Not recommended for high-stakes applications (e.g., mental health assessment)
- **Mitigation**: Fine-tuned on task-specific data for better contextual representation

### Caveats and Recommendations
- **Testing**: More thorough evaluation needed for underrepresented classes
- **Subjectivity**: Emotion interpretation varies across individuals, introducing inconsistencies
- **Fine-Tuning**: Additional domain-specific tuning required for sensitive applications

---

## 🛠️ Technologies Used

**NLP & Deep Learning:**
- Hugging Face Transformers (DistilRoBERTa, mBART)
- PyTorch
- Scikit-learn

**Speech Processing:**
- AssemblyAI API for speech-to-text

**Data Processing:**
- pandas, NumPy
- Python standard libraries

**Explainability:**
- SHAP, LIME (Task 9)

---

## 👥 Team Collaboration

This project was developed collaboratively by a 3-person team as part of Block C at Breda University of Applied Sciences. The work demonstrates effective team coordination across multiple NLP tasks, model development, and pipeline integration.

---

## 📈 Future Improvements

- Address class imbalance for rare emotions (disgust, fear)
- Expand training data for underrepresented classes
- Implement multi-label classification for mixed-emotion sentences
- Optimize translation model for real-time processing
- Add support for more languages beyond current 6+

---

## 📄 License

This project uses models under Apache 2.0 license. See individual model cards for specific licensing details.
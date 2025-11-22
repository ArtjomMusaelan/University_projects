Content Intelligence Emotion Recognition - Phase 2: Production Deployment

**Enterprise MLOps Platform for Multi-Modal Emotion Analysis**

[![Azure](https://img.shields.io/badge/Azure-ML%20Deployed-blue.svg)](https://azure.microsoft.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)](https://fastapi.tiangolo.com/)
[![Airflow](https://img.shields.io/badge/Airflow-Orchestrated-red.svg)](https://airflow.apache.org/)

---

## Overview

Production deployment of the emotion detection models from [Phase 1](https://github.com/ArtjomMusaelan/University_projects/tree/main/NLP_ContentIntelligenceAgency). Transforms research prototypes into a scalable, cloud-deployed MLOps platform with automated pipelines, multiple deployment strategies, and comprehensive monitoring.

**Client:** Content Intelligence (outstanding content)  
**Institution:** Breda University of Applied Sciences  
**Duration:** 8-week Agile sprint (Block D, Year 2, 2025)

### Key Capabilities

- **Audio Processing:** YouTube/MP3 → Romanian transcription → English translation → Emotion analysis
- **Text Analysis:** 7 core + 19 fine-grained emotions with confidence scores
- **Batch Processing:** Large-scale CSV emotion analysis
- **Automated Pipelines:** Weekly data preprocessing via Airflow
- **Multiple Interfaces:** REST API, Streamlit web UI, Azure ML endpoints
- **Feedback Loop:** User feedback → Azure ML → Continuous improvement

---

## Architecture

```
┌─────────────┐
│   Users     │
└──────┬──────┘
       │
┌──────▼───────────────────────────────────────────────────┐
│            Production Environment                        │
│                                                           │
│  ┌──────────────┐      ┌──────────────┐                 │
│  │ Streamlit UI │      │  FastAPI     │                 │
│  │  (Frontend)  │─────▶│   (API)      │                 │
│  └──────────────┘      └──────┬───────┘                 │
│                                │                          │
│  ┌────────────────────────────▼──────────────────────┐  │
│  │         NLP Model Pipeline                        │  │
│  │  Whisper → LLaMA 3.3 → OPUS-MT → CamemBERT       │  │
│  │  (Audio)   (Correction) (Translation) (Emotion)   │  │
│  └────────────────────────────────────────────────────┘ │
│         │                    │                           │
│  ┌──────▼──────┐      ┌─────▼─────┐                    │
│  │   Airflow   │      │  Training  │                    │
│  │  Pipeline   │      │    Jobs    │                    │
│  │  (Weekly)   │      │ (Azure ML) │                    │
│  └─────────────┘      └────────────┘                    │
└───────────────────────────────────────────────────────────┘
                       │
          ┌────────────▼──────────────┐
          │    Azure ML Workspace     │
          │  - Environments (v4, v5)  │
          │  - Endpoints (Kubernetes) │
          │  - Model Registry         │
          │  - Blue-Green Deployment  │
          │  - Feedback Datasets      │
          └───────────────────────────┘
```

---

## Technical Achievements

### Performance Metrics
- **44.7 RPS** sustained throughput (1,978 requests tested)
- **117ms** average response time (90ms median, 270ms 99th percentile)
- **99.9% uptime** (1 failure in 1,978 requests)
- **98.78% confidence** for joy detection

### ML Pipeline
1. **Whisper (large):** Romanian/English speech-to-text
2. **LLaMA 3.3:** Spelling correction with 200 parallel workers
3. **OPUS-MT:** Romanian → English translation
4. **CamemBERT:** Emotion classification (7 core + 19 fine-grained)
5. **TextBlob:** Sentiment intensity scoring

### Deployment Infrastructure
- **Blue-Green Strategy:** Zero-downtime updates with 80/20 traffic shifting
- **Docker Compose:** Multi-service orchestration with health checks
- **Azure ML:** Kubernetes endpoints on `adsai-lambda-0/2` compute
- **Airflow DAG:** Weekly automated data preprocessing
- **GitHub Actions:** CI/CD with linting, testing, Docker builds

---

## Project Structure

```
NLP_ContentIntelligenceAgency(Phase2)/
│
├── .github/workflows/              # CI/CD automation
│   ├── docker.yaml
│   ├── lint.yaml
│   └── test.yaml
│
├── airflow/dags/                   # Workflow orchestration
│   ├── emotion_preprocessing_dag.py
│   └── text_preprocessing.py
│
├── api/                            # Basic FastAPI with load testing
│   ├── Dockerfile
│   ├── FastAPI.py                  # 5 endpoints
│   ├── locust.py
│   ├── locust_logs.yaml            # 1,978 request results
│   └── requirements.txt
│
├── azure_env/                      # Azure ML environment setup
│   ├── config.json                 # Workspace: NLP8-2025
│   ├── conda.yaml                  # Environment: nlp8-env3
│   ├── register_env.py
│   └── README.md
│
├── cloud_deployment/
│   ├── blue_green_deployment/      # Zero-downtime strategy
│   │   ├── create_green_deployment.py
│   │   ├── rollback_to_blue.py
│   │   ├── switch_traffic_80_20.py
│   │   ├── switch_traffic_to_green.py
│   │   ├── score.py
│   │   └── test_request.py
│   └── simple_deployment/          # Basic Azure ML deployment
│       ├── create_endpoint.py
│       └── get_logs.py
│
├── data/                           # Dataset management
│   ├── upload_temp/                # Train/val/test splits
│   ├── temp_download/
│   └── split_register.ipynb
│
├── deploy_azure/                   # Streamlit + FastAPI deployment
│   ├── yaml/
│   │   ├── emotion-backend.yaml
│   │   └── emotion-frontend.yaml
│   ├── docker-compose.yaml
│   ├── Dockerfile
│   ├── Dockerfile.streamlit
│   ├── main.py
│   └── streamlit_app.py            # 4-tab UI
│
├── docs/                           # Project documentation
│   ├── Local + azure deployment.pdf
│   ├── Poster_NLP8.pdf
│   ├── Presentation_NLP8.pdf
│   ├── Project Roadmap.pdf
│   └── [Sphinx documentation]
│
├── emotion_classification_api/     # Enhanced API with monitoring
│   ├── utils/
│   │   ├── logger_feedback.py      # Latency tracking, feedback
│   │   └── __init__.py
│   ├── run.py                      # Monitored predictions
│   └── test_api.py
│
├── fastapi/                        # Alternative implementation
│   ├── Dockerfile
│   ├── FastAPI.py
│   └── requirements.txt
│
├── FastApi_deploying/              # Full production stack
│   ├── docker-compose.yml          # Backend + Frontend orchestration
│   ├── Dockerfile
│   ├── Dockerfile.streamlit
│   ├── main.py                     # 7 endpoints + Azure ML feedback
│   ├── streamlit_app.py            # Professional UI
│   └── requirements.txt
│
├── financial_costs/
│   └── cost_evaluation.pdf         # Blue-green justification, optimization
│
├── training_pipeline/              # Automated model training
│   ├── version1/                   # Initial: 3 epochs, diagnostics
│   │   ├── script_folder/
│   │   │   ├── model_diagnostics.py
│   │   │   └── train_script_v1.py
│   │   ├── azure_pipeline_v1.py
│   │   ├── register_model.py
│   │   └── AzureCLI
│   └── version2/                   # Optimized: 1 epoch, lambda-0
│       ├── azure_pipeline.py
│       └── train_script.py
│
├── src/                            # Source code (restructuring)
│
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## API Endpoints

### Production FastAPI

**Text Prediction:**
```bash
POST /predict
{"text": "I love this project!"}
# Returns: emotion predictions with confidence scores
```

**CSV Batch Processing:**
```bash
POST /upload-csv
# Upload CSV with 'text' column
# Returns: predictions saved to /app/output/
```

**Download Results:**
```bash
GET /download-csv
# Downloads batch prediction results
```

**User Feedback:**
```bash
POST /feedback-csv
# Uploads feedback to Azure ML for retraining
```

**YouTube Transcript:**
```bash
POST /youtube-transcript?url=YOUTUBE_URL
# Downloads audio → transcribes → saves CSV
```

**Audio Upload:**
```bash
POST /upload-audio-transcript
# Transcribes MP3/WAV/M4A → saves CSV
```

**Health Check:**
```bash
GET /health
# Container orchestration health monitoring
```

---

## Streamlit Web Interface

**4-Tab User Interface:**

1. **Predict Emotion**
   - Text input with real-time prediction
   - 3 visualizations: bar chart, pie chart, horizontal distribution
   - Confidence scores for all 7 emotions

2. **Upload CSV**
   - Bulk emotion analysis
   - Results preview with DataFrame
   - Emotion distribution chart
   - Good/Bad feedback buttons → Azure ML upload
   - Download predictions

3. **YouTube Transcript**
   - Enter YouTube URL
   - Generate transcript CSV
   - Preview and download

4. **Upload Audio**
   - MP3/WAV/M4A support
   - Whisper transcription
   - Preview and download

---

## MLOps Workflows

### Airflow Pipeline
- **Schedule:** Weekly (`@weekly`)
- **Tasks:** Download from Azure Blob → Preprocess → Register dataset
- **Authentication:** Service Principal via Airflow connections

### Training Pipeline
- **Compute:** adsai-lambda-0/2
- **Environment:** nlp8-env2 (v4, v5)
- **Dataset:** Registered train/val splits
- **Metrics:** Accuracy, F1, precision, recall logged to Azure ML
- **Model Registry:** Auto-registration after training

### Blue-Green Deployment
1. Deploy green alongside blue
2. Test green with `test_request.py`
3. Shift 80/20 traffic for validation
4. Complete cutover to 100% green
5. Rollback capability if needed

**Cost Justification:**
- Only one deployment serves traffic (no duplicate costs)
- Safe testing without Shadow/Canary overhead
- Avoids Big Bang risk while minimizing expenses

---

## Technology Stack

**ML/AI:** CamemBERT, RoBERTa, Whisper, OPUS-MT, LLaMA 3.3, TextBlob  
**Backend:** FastAPI, Uvicorn, Pydantic  
**MLOps:** Airflow, Azure ML, Docker, GitHub Actions  
**Data:** pandas, PyTorch, TensorFlow, scikit-learn, Optuna  
**Frontend:** Streamlit, Plotly  
**Testing:** Locust, pytest  
**Cloud:** Azure ML SDK, Azure Blob Storage, Azure Identity  
**Audio:** yt-dlp, ffmpeg, openai-whisper, NLTK

---

## Key Metrics

| Metric | Value |
|--------|-------|
| RPS (sustained) | 44.7 requests/second |
| Response time (avg) | 117ms |
| Uptime | 99.9% |
| Parallel workers | 200 (LLaMA correction) |
| Core emotions | 7 classes |
| Fine-grained emotions | 19 classes |
| Load test requests | 1,978 |

---

## MLOps Maturity

**Level 4 - Full Automation:**

- Automated data pipelines (Airflow)  
- CI/CD (GitHub Actions)  
- Model versioning (Azure ML)  
- Monitoring & logging  
- User feedback loops  
- Blue-green deployment  
- Cost optimization 
- Comprehensive testing  

---

## Related Projects

**Phase 1:** [Research & Model Development](https://github.com/ArtjomMusaelan/University_projects/tree/main/NLP_ContentIntelligenceAgency)
- Focus: French emotion detection, explainable AI
- Achievement: 63% accuracy, LIME + Grad-CAM visualization

---

**Production-ready AI with enterprise MLOps practices.**

*Last Updated: July 2025*
# Data Science & AI Portfolio

**Artjom Musaelans**  
Data Science & Artificial Intelligence Student  
Breda University of Applied Sciences

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/artem-musaelan-50a54a327/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/ArtjomMusaelan)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:artjom2004.lv@gmail.com)

---

## About Me

I'm a third-year Data Science and AI student at Breda University of Applied Sciences, passionate about applying machine learning and data analytics to solve real-world problems. My portfolio demonstrates progression from foundational data visualization and traditional ML to advanced deep learning, NLP, and enterprise-scale MLOps deployment.

Throughout my studies, I've worked on diverse projects spanning healthcare analytics, computer vision, natural language processing, and production system deployment. I've collaborated with real clients including NPEC (Netherlands Plant Eco-phenotyping Centre), Content Intelligence Agency, ANWB, and DigiWerkplaats, delivering solutions that address genuine business challenges.

My work emphasizes not just technical excellence but also responsible AI practices, including explainability, fairness analysis, GDPR compliance, and adherence to EU AI Act requirements.

---

## Core Competencies

### Machine Learning & Deep Learning
- **Supervised Learning:** Classification & Regression (Linear Regression, Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, XGBoost)
- **Computer Vision:** CNNs, U-Net, Semantic Segmentation, Instance Segmentation, Image Classification
- **Natural Language Processing:** Transformers (BERT, DistilRoBERTa, mBART), LSTM, RNN, Feature Engineering (POS tagging, TF-IDF, Word Embeddings)
- **Model Evaluation:** Cross-validation, Hyperparameter Tuning (GridSearchCV), Custom Metrics Development

### MLOps & Production Deployment
- **Cloud Platforms:** Azure ML (Kubernetes Endpoints, Model Registry, Managed Environments)
- **DevOps:** Docker Containerization, CI/CD Pipelines (GitHub Actions), Blue-Green Deployment
- **Orchestration:** Apache Airflow DAGs, Automated Data Pipelines
- **API Development:** FastAPI with 7+ Endpoints, Load Testing 
- **Monitoring:** Health Checks, Logging, Performance Metrics, User Feedback Loops

### Data Engineering & Analysis
- **Programming:** Python (pandas, NumPy, scikit-learn, TensorFlow, PyTorch)
- **Data Visualization:** Power BI (DAX, Power Query), Matplotlib, Plotly, Streamlit
- **Traditional CV:** OpenCV, scikit-image, Morphological Operations
- **Graph Algorithms:** NetworkX, Dijkstra's Algorithm, Skeleton Analysis
- **Research Methods:** Mixed-methods Research, Statistical Analysis, Qualitative Coding

### Specialized Skills
- **Explainable AI:** LIME, SHAP, Layer-wise Relevance Propagation, Conservative Propagation, Grad-CAM
- **Speech Processing:** AssemblyAI, Whisper, Word Error Rate Analysis
- **Web Scraping:** Selenium Automation
- **Ethics & Compliance:** EU AI Act Classification, GDPR, Research Ethics Board Approval, FAIR Data Principles

---

## Projects

### Year 1 - Foundation Building

#### [Tobacco Mortality Analysis Dashboard](./SDG3_Smoking_Mortality)
**Block A | Power BI | UN SDG 3 Support**

Interactive dashboard analyzing 30 years (1990-2019) of global tobacco-related mortality. Discovered strong positive correlations (r = 0.84-0.99) between tobacco use and mortality across age groups. Created 4-tab system with geographic maps, trend analysis, and correlation views using Power Query ETL and DAX calculations.

**Achievement:** 0.99 correlation for ages 15-49 and 50-69, 0.98 for 70+  
**Technologies:** Power BI, Power Query, DAX  
**Impact:** Evidence-based visualization for health organizations and policymakers

---

#### [NAC Player Analytics & Goal Prediction](./Goal_Prediction_NAC)
**Block B | Machine Learning | Sports Analytics**

ML pipeline processing 16,535+ player records with 116 features. Achieved 99.03% prediction accuracy (R²=0.9903) using Linear Regression after rigorous data cleaning and feature engineering. Compared 6+ algorithms, created 15+ visualizations, implemented custom gradient descent, and identified xG as strongest predictor.

**Achievement:** 99.03% accuracy with custom gradient descent implementation  
**Technologies:** Python, scikit-learn, pandas, matplotlib, NumPy  
**Impact:** Recruitment decision support for NAC Breda football club

---

#### [Natural Disaster Image Classifier](./Classifer_of_natural_disasters)
**Block C | Deep Learning | Computer Vision**

CNN system achieving 98.3% validation accuracy across 2 disaster classes (floods, wildfires). Scraped 300+ images, implemented 4 progressive CNN architectures, integrated explainable AI (LIME, KernelSHAP), conducted fairness analysis, and performed human-centered design with A/B testing.

**Achievement:** 98.3% validation accuracy, 87.5% test accuracy  
**Technologies:** TensorFlow/Keras, Selenium, LIME, KernelSHAP  
**Impact:** Early warning system for emergency response agencies

---

#### [Road Safety Risk Prediction](./ANWB)
**Block D | Production ML | Team Project**

Street-level safety prediction system for ANWB (Dutch automobile association). As Data Scientist on 5-person team, built XGBoost models integrating 2017-2023 incident data with KNMI weather. Deployed Streamlit application classifying Breda streets as high/low risk with complete EU AI Act compliance.

**Achievement:** Production deployment with EU AI Act compliance  
**Technologies:** Python, XGBoost, scikit-learn, Streamlit, SQL  
**Impact:** Proactive route planning for ANWB drivers  
**Role:** Data Scientist (model development & deployment) | **Team:** 5 members

---

### Year 2 - Advanced Applications & Production

#### [Hotel Cybersecurity Research](./Policy_Research_for_SMEs)
**Block A | Academic Research | Policy Development**

Mixed-methods research for DigiWerkplaats investigating employee awareness impact on small hotel security. Combined Qualtrics surveys with structured interviews, discovering moderate correlation (r = 0.456) between training frequency and security adherence. Delivered policy paper recommending quarterly training with complete GDPR compliance and ethics approval.

**Achievement:** Published research with actionable policy recommendations  
**Technologies:** Python, scikit-learn, Jupyter, Qualtrics  
**Impact:** Evidence-based cybersecurity training for hospitality SMEs  
**Team:** 4 members | **Partner:** DigiWerkplaats Breda

---

#### [Plant Root Segmentation & Phenotyping](./Root_Segmentation)
**Block B | Computer Vision | Agricultural AI**

8-stage CV pipeline for NPEC automating Arabidopsis thaliana root analysis. Combined traditional CV (Petri dish detection ±30px accuracy, plant instance segmentation), U-Net deep learning (82.4% F1, 1.9M parameters), and graph-based RSA extraction (NetworkX, Dijkstra's algorithm). Enables robotic inoculation of 10,000+ seedlings.

**Achievement:** 82.4% F1 score (exceeded 80% target)  
**Technologies:** TensorFlow, OpenCV, NetworkX, scikit-image  
**Impact:** High-throughput phenotyping automation for plant research  
**Client:** NPEC (Netherlands Plant Eco-phenotyping Centre)

---

#### [Emotion Classification Pipeline](./NLP_ContentIntelligenceAgency)
**Block C | NLP | Multi-Language System**

Production NLP pipeline for Content Intelligence Agency automating emotion detection in unscripted TV shows. Integrated AssemblyAI/Whisper transcription, fine-tuned mBART translation, and DistilRoBERTa classifier (71% weighted F1 on 86,224 samples). Implemented Word Error Rate analysis, Conservative Propagation XAI, and comprehensive error analysis across 12+ tasks.

**Achievement:** 71% weighted F1 with advanced XAI (Conservative Propagation)  
**Technologies:** PyTorch, Transformers, SpaCy, NLTK, AssemblyAI, Gensim  
**Impact:** Automated emotion analysis for media content across multiple languages  
**Team:** 3 members | **Client:** Content Intelligence Agency

---

#### [MLOps Production Deployment](./NLP_ContentIntelligenceAgency_Phase2)
**Block D | Cloud Infrastructure | Enterprise Systems**

Transformed Phase 1 research into enterprise Azure ML platform. Achieved 44.7 RPS throughput, 117ms average response time, 99.9% uptime (1,978 load test requests). Implemented Docker containerization, FastAPI (7 endpoints), 4-tab Streamlit UI, Airflow weekly pipelines, GitHub Actions CI/CD, and blue-green deployment with user feedback loops.

**Achievement:** 99.9% uptime, Level 4 MLOps maturity (full automation)  
**Technologies:** Azure ML, Docker, FastAPI, Airflow, GitHub Actions, Streamlit  
**Impact:** Scalable emotion analysis platform for Content Intelligence Agency  
**Team:** 5 members (Agile sprint)

---

## Skills Development Journey

### Year 1 - Foundation Building
**Focus:** Data visualization, statistical analysis, traditional ML

**Projects:** Tobacco Dashboard, NAC Analytics, Disaster Classifier, ANWB Safety

**Skills Acquired:**
- Power BI dashboard design and DAX measures
- Python for data science (pandas, NumPy, scikit-learn etc.)
- Machine learning algorithms and model comparison
- Statistical correlation and regression analysis
- CNN architecture and image classification
- Web scraping and data collection
- Team collaboration in production environment

---

### Year 2 - Advanced Applications & Production
**Focus:** Deep learning, NLP, research, enterprise deployment

**Projects:** Cybersecurity Research, Root Segmentation, Emotion Classification, MLOps Deployment

**Skills Acquired:**
- Academic research methodology (mixed-methods)
- Advanced computer vision (U-Net, semantic segmentation)
- Graph algorithms for complex analysis
- Natural language processing (Transformers, fine-tuning)
- Speech-to-text and machine translation
- Explainable AI techniques (LIME, SHAP, LRP)
- Enterprise MLOps and cloud deployment
- CI/CD, Docker, and production systems
- GDPR and EU AI Act compliance

---

## Technical Skills Summary

### Programming & Frameworks
**Expert:** Python (pandas, NumPy, scikit-learn, TensorFlow, PyTorch)  
**Proficient:** SQL, DAX, Docker, Git  
**ML Frameworks:** Keras, Hugging Face Transformers, XGBoost  
**CV Libraries:** OpenCV, scikit-image, PIL  
**NLP Tools:** SpaCy, NLTK, Gensim, AssemblyAI, Whisper

### Cloud & Infrastructure
**Cloud:** Azure ML (Endpoints, Registry, Compute, Environments)  
**DevOps:** Docker, GitHub Actions, Airflow  
**APIs:** FastAPI, REST API Development  
**Testing:** Locust, pytest  
**Deployment:** Blue-Green Strategy, Kubernetes

### Data & Visualization
**BI Tools:** Power BI (Power Query, DAX)  
**Visualization:** Matplotlib, Plotly, Seaborn, Streamlit  
**Data Processing:** ETL Pipelines, Data Cleaning, Feature Engineering  
**Research:** Qualtrics, Statistical Testing, Thematic Analysis

---

## Portfolio Impact

| Category | Metric |
|----------|--------|
| **Total Projects** | 8 completed projects |
| **Team Collaborations** | 4 team projects (3-5 person teams) |
| **Real Clients** | 4 organizations (NPEC, Content Intelligence Agency, ANWB, DigiWerkplaats) |
| **Highest ML Accuracy** | 99.03% (NAC Player Analytics) |
| **Production Uptime** | 99.9% (MLOps Deployment) |
| **Largest Training Dataset** | 86,224 samples (Emotion Classification) |
| **Best F1 Score (CV)** | 82.4% (Plant Root Segmentation) |
| **Best F1 Score (NLP)** | 71% weighted (Emotion Classification) |
| **Published Research** | 1 (Hotel Cybersecurity Policy) |

---

## Academic Background

**Institution:** Breda University of Applied Sciences  
**Program:** Data Science & Artificial Intelligence (BSc)  
**Current Status:** Year 3  
**Expected Graduation:** 2026

### Blocks Completed

**Year 1:**
- Block A: Tobacco Mortality Dashboard (Power BI, Public Health)
- Block B: NAC Player Analytics (ML, Football)
- Block C: Natural Disaster Classifier (Deep Learning, CV)
- Block D: ANWB Road Safety (Production ML, Team Project)

**Year 2:**
- Block A: Cybersecurity Research (Mixed-methods, Policy)
- Block B: Plant Root Segmentation (U-Net, Graph Algorithms)
- Block C: Emotion Classification (NLP, Transformers)
- Block D: MLOps Deployment (Azure ML, Production)

---

## Key Achievements

### Technical Excellence
- **99.03% Prediction Accuracy** - NAC Player Analytics (Linear Regression)
- **98.3% Validation Accuracy** - Natural Disaster Classifier (CNN)
- **82.4% F1 Score** - Plant Root Segmentation (exceeded 80% target)
- **71% Weighted F1** - Emotion Classification (7 emotion categories)
- **99.9% Uptime** - Production MLOps Platform
- **44.7 RPS** - Sustained API Throughput

### Research & Innovation
- Published cybersecurity research with evidence-based policy recommendations
- Implemented cutting-edge XAI (Conservative Propagation for Transformers)
- Custom algorithm development (gradient descent, graph traversal for RSA)
- Strong correlation discoveries in health analytics (r = 0.84-0.99)

### Professional Skills
- Data Scientist role in 5-person Agile team (ANWB project)
- Multi-language NLP systems (6+ languages supported)
- Enterprise cloud deployment (Azure ML Level 4 maturity)
- Complete ethics compliance (GDPR, EU AI Act, Research Ethics)

---

## Repository Structure

```
portfolio_projects/
│
├── README.md                                    # This file - Portfolio overview
│
├── SDG3_Smoking_Mortality/                     # Year 1 - Block A
│   ├── README.md
│   ├── SDGIndicatorsDashboard_Data_visualization.pbix
│   └── Certificates/
│
├── Goal_Prediction_NAC/                        # Year 1 - Block B
│   ├── README.md
│   ├── Final_Deliverable.ipynb
│   ├── Final_Report.pdf
│   └── Certificates/
│
├── Classifer_of_natural_disasters/             # Year 1 - Block C
│   ├── README.md
│   ├── Classifer_of_natural_disasters.ipynb
│   ├── Image Classificator ver.B.fig
│   └── Project-Presentation.pptx
│
├── ANWB/                                       # Year 1 - Block D
│   ├── README.md
│   └── [Team repository structure]
│
├── Policy_Research_for_SMEs/                   # Year 2 - Block A
│   ├── README.md
│   ├── Policy_paper_Cybersecurity.pdf
│   ├── Research_proposal.pdf
│   ├── Poster.png
│   ├── Data/
│   │   ├── cleaned_survey.xlsx
│   │   ├── Interview(audio)/
│   │   └── Interview(transcript)/
│   └── DMP/
│       ├── Codebook.md
│       ├── NWO_DMP_Data_Mangament_plan.pdf
│       └── [Ethics documentation]
│
├── Root_Segmentation/                          # Year 2 - Block B
│   ├── README.md
│   ├── final_pipeline/
│   ├── model_training/
│   ├── root_system_architecture_extraction/
│   ├── individual_root_segmentation/
│   ├── petri_dish_detection_extraction/
│   ├── plant_instance_segmentation/
│   ├── image_annotation/
│   ├── data_preparation/
│   └── models/
│
├── NLP_ContentIntelligenceAgency/              # Year 2 - Block C
│   ├── README.md
│   ├── .gitignore
│   └── Deliverables/
│       ├── requirements.txt
│       ├── Task2 - Speech to Text/
│       ├── Task3 - WER rate/
│       ├── Task4 - NLP features/
│       ├── Task5 - Model Iterations/
│       ├── Task6 - Machine Translation/
│       ├── Task7 - Prompt Engineering/
│       ├── Task8 - Error Analysis/
│       ├── Task9 - Explainable AI (XAI)/
│       ├── Task10 - Model Card/
│       ├── Task11 - Pipeline/
│       ├── Task12 - Presentation/
│       └── SWOT analysis/
│
└── NLP_ContentIntelligenceAgency_Phase2/       # Year 2 - Block D
    ├── README.md
    ├── .github/workflows/
    ├── airflow/dags/
    ├── azure_env/
    ├── cloud_deployment/
    ├── deploy_azure/
    ├── FastApi_deploying/
    ├── training_pipeline/
    ├── docs/
    └── financial_costs/
```

---

## Featured Projects

### Best ML Performance
**NAC Player Analytics** - 99.03% prediction accuracy

### Most Production-Ready
**MLOps Deployment** - 99.9% uptime, 44.7 RPS, Level 4 automation

### Most Technically Complex
**Plant Root Segmentation** - 8-stage pipeline combining traditional CV, deep learning, and graph algorithms

### Most Comprehensive
**Emotion Classification** - 12+ tasks from speech-to-text through XAI to production deployment

### Best Research
**Cybersecurity Study** - Published mixed-methods research with policy impact

---

## Project Categorization

### By Technical Domain

**Computer Vision**
- Natural Disaster Classifier (CNN, 98.3% accuracy)
- Plant Root Segmentation (U-Net, 82.4% F1)

**Natural Language Processing**
- Emotion Classification Pipeline (DistilRoBERTa, 71% F1)
- MLOps Production Deployment (Azure ML, 99.9% uptime)

**Traditional Machine Learning**
- NAC Player Analytics (Linear Regression, 99% accuracy)
- Road Safety Prediction (XGBoost, production app)

**Data Analysis & Visualization**
- Tobacco Mortality Dashboard (Power BI, UN SDG 3)
- Cybersecurity Research (Mixed-methods, r = 0.456)

---

### By Application Area

**Healthcare & Life Sciences**
- Tobacco Mortality Analysis (Public Health Policy)
- Plant Root Segmentation (Agricultural Research)

**Business & Analytics**
- NAC Player Analytics (Sports Recruitment)
- Road Safety Prediction (Automotive Safety)
- Cybersecurity Research (Hospitality Security)

**Media & Technology**
- Emotion Classification (TV Content Analysis)
- MLOps Deployment (Media Analysis Platform)
- Disaster Classifier (Emergency Response)

---

## What Makes This Portfolio Stand Out

### 1. **Clear Progression**
Demonstrates growth from basic visualization (Year 1) → advanced deep learning (Year 2) → enterprise production systems (Year 2)

### 2. **Real-World Impact**
All projects solved actual problems for real clients and organizations, not just academic exercises

### 3. **Full-Stack ML Expertise**
End-to-end capabilities: data collection → preprocessing → modeling → deployment → monitoring

### 4. **Responsible AI Focus**
Consistent emphasis on ethics, explainability, fairness, and legal compliance across all projects

### 5. **Diverse Applications**
Eight projects spanning seven different domains, showing adaptability and broad technical knowledge

### 6. **Team Collaboration**
Five team projects with clear role definition, demonstrating professional collaboration skills

### 7. **Production Experience**
Multiple deployed systems with real users, performance metrics, and continuous improvement

---

## Contact Information

**Email:** artjom2004.lv@gmail.com 
**LinkedIn:** [linkedin.com/in/artjom-musaelans](https://www.linkedin.com/in/artem-musaelan-50a54a327/)  
**GitHub:** [github.com/artjom-musaelans](https://github.com/ArtjomMusaelan)  
**Location:** Breda, Netherlands  
**University:** Breda University of Applied Sciences

**Status:** Seeking Data Science internships and opportunities

---

## Career Interests

### Immediate Goals
- Data Science internships across any domain (open to all opportunities)
- Applied ML engineering roles with production deployment
- Data engineering and MLOps positions
- Any roles involving data analysis, AI development, or system deployment

### Long-term Vision
- Specialize in cutting-edge AI applications (healthcare, NLP, MLOps, or emerging fields)
- Contribute to impactful AI solutions addressing real-world challenges
- Bridge gap between research and production-ready systems
- Continuous learning and adaptation to new technologies

### Areas of Interest
- **Open to All Domains:** Healthcare, Finance, Agriculture, Media, Retail, Manufacturing, etc.
- **Data-Related Work:** Data engineering, analysis, visualization, pipeline development
- **AI & ML:** Model development, deep learning, NLP, computer vision
- **Deployment & Production:** MLOps, cloud infrastructure, API development, DevOps
- **Responsible AI:** Ethics, explainability, fairness, compliance

**Current Status:** Actively seeking internship opportunities in a data/AI-related field

---

## Documentation Standards

All projects in this portfolio follow consistent documentation standards:

- README with clear structure  
- Technical details and architecture explanations  
- Performance metrics and evaluation results  
- Technology stack and dependencies  
- Setup instructions where applicable  
- Key findings and business impact  
- Proper attribution and citations

---

## Quick Links

| Project | Domain | Year | Key Metric | Link |
|---------|--------|------|------------|------|
| Tobacco Dashboard | Health Analytics | 1 | r = 0.99 correlation | [View →](./SDG3_Smoking_Mortality) |
| NAC Analytics | Sports ML | 1 | 99.03% accuracy | [View →](./Goal_Prediction_NAC) |
| Disaster Classifier | Computer Vision | 1 | 98.3% accuracy | [View →](./Classifer_of_natural_disasters) |
| ANWB Safety | Production ML | 1 | EU AI compliant | [View →](./ANWB) |
| Cybersecurity | Research | 2 | Published paper | [View →](./Policy_Research_for_SMEs) |
| Root Segmentation | CV + Graphs | 2 | 82.4% F1 | [View →](./Root_Segmentation) |
| Emotion NLP | Transformers | 2 | 71% F1 | [View →](./NLP_ContentIntelligenceAgency) |
| MLOps Platform | Cloud Deploy | 2 | 99.9% uptime | [View →](./NLP_ContentIntelligenceAgency_Phase2) |

---


## Acknowledgments

**Academic Institution:** Breda University of Applied Sciences - Faculty of Data Science & AI

**Client Organizations:**
- NPEC (Netherlands Plant Eco-phenotyping Centre)
- Content Intelligence Agency
- ANWB (Royal Dutch Touring Club)
- DigiWerkplaats Breda

**Mentors & Instructors:** BUas faculty for guidance and support throughout all projects

---

**Last Updated:** November 2025  
**Projects Completed:** 8/8  
**Status:** Active Portfolio - Available for Opportunities

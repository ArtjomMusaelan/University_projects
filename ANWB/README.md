# Safer Roads in Breda

A machine learning application predicting street-level road safety risk to enhance driver awareness and reduce accidents in Breda.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![EU AI Act](https://img.shields.io/badge/EU_AI_Act-Limited_Risk-green.svg)

## Project Overview

**Safer Roads in Breda** is an AI-powered driver safety application developed for ANWB that predicts street-level risk based on historical incident data, weather conditions, and temporal patterns. The application provides drivers with actionable insights to improve road safety and reduce accidents in the municipality of Breda.

### Problem Statement

Despite national efforts to improve road safety, existing navigation applications lack localized, street-level risk information. Even low-traffic streets can be dangerous due to factors like weather conditions, vegetation, and unexpected pedestrian/cyclist activity. Traditional methods are reactive and lack predictive accuracy.

### Solution

A machine learning application that:
- Predicts high-risk vs. low-risk streets based on multiple data sources
- Integrates weather conditions with historical incident patterns
- Provides street-level safety recommendations to drivers
- Enables proactive route planning and increased driver awareness
- Complies with EU AI Act (Limited Risk) and GDPR regulations

## Key Features

**Street-Level Risk Prediction** - Classify roads as high-risk or low-risk  
**Multi-Source Data Integration** - ANWB incidents + KNMI weather + accident data  
**Interactive Application** - Streamlit-based user interface  
**Daily Updates** - Predictions based on current date and weather  
**EU AI Act Compliant** - Limited Risk classification with transparency  
**GDPR Compliant** - Privacy-preserving data handling  

## Repository Structure

```
safer-roads-breda/
│
├── certificates/             # Academic certifications
├── data_cleaning/            # Data preprocessing scripts
├── eda/                      # Exploratory data analysis notebooks
├── modelling/                # Model training and evaluation
├── streamlit_app/            # Deployment application
├── proposal_blockd_final.pdf # Project proposal document
└── README.md                 # This file
```

## Datasets

### Primary Data Sources

**1. ANWB Safe Driving Dataset (2017-2023)**
- Incident records for Breda municipality
- **Categories**: Speed, Harsh Cornering, Harsh Accelerating, Harsh Braking
- **Key Variables**:
  - `road_name` - Street where incident occurred
  - `category` - Type of driving incident
  - `incident_severity` - Severity scale
  - `maxwaarde` - Max speed (km/h) or g-force value
  - `latitude/longitude` - Location coordinates
  - `dtg` - Timestamp

**2. KNMI Weather Data**
- **Components**: Wind, Precipitation, Temperature
- **Coverage**: Breda-wide (not street-specific)
- **Key Variables**:
  - `RI_REGENM_10` - Precipitation measurements
  - `dtg` - Timestamp for weather alignment

**3. Bron Accidents Dataset**
- Supplementary accident data
- Road condition information
- Timestamp and location coordinates

## Methodology

### Machine Learning Process

**1. Data Cleaning** (data_cleaning/)
- Duplicate removal
- Outlier filtering (IQR method, Q < 0.25 and Q > 0.75)
- Structural error fixes (naming conventions, typos, capitalization)
- Missing value handling (conversion to NaN)
- Column filtering (removal of low-value features)

**2. Data Preprocessing**
- Standard scaling for numerical features
- Format transformation
- Binary column conversion for balanced distribution
- Multi-dataset merging (ANWB + KNMI + Bron)
- Data conflict resolution

**3. Exploratory Data Analysis** (eda/)
- Pattern and relationship discovery
- Value distribution plotting
- Correlation heatmap analysis
- Feature importance identification

**4. Feature Engineering**
- Weather condition categorization
- Temporal feature extraction
- Location-based features
- Incident severity grouping

**5. Model Development** (modelling/)
- **Primary Model**: XGBoost Classifier
- **Rationale**: Computational efficiency and strong performance
- **Approach**: Gradient boosting with sequential error correction
- Multiple model iterations (with/without coordinates)

**6. Model Evaluation**
- **Metrics**: RMSE, Recall, MAE
- Validation set performance testing
- Overfitting detection

**7. Optimization**
- GridSearchCV for hyperparameter tuning
- Systematic parameter exploration
- Model iteration and refinement

**8. Deployment** (streamlit_app/)
- Streamlit web application
- Interactive user interface
- Real-time risk prediction

### Development Approach

**Proof-of-Concept (POC) Model:**
- Try-and-test approach for feasibility validation
- No customer journeys (internal reference)
- Iterative development loop:
  1. Model design (add features)
  2. Training
  3. Error analysis
  4. Refinement

## Legal & Ethical Framework

### EU AI Act Classification

**Risk Level:** Limited Risk

**Rationale:**
- AI system is not a safety component of a product
- Provides advisory predictions, not autonomous control
- Final driving decisions remain with the user
- Does not directly put lives at risk

### Legal Obligations

**Required for Limited Risk Systems:**
- **Transparency**: Users informed about AI-generated predictions
- **Documentation**: Risk assessment and classification reasoning
- **EU Database Registration**: System and provider registration (Article 49)
- **Union Harmonisation**: Compliance with testing and reporting (Article 8)
- **Systemic Risk Assessment**: Monitoring per Article 51-52
- **Code of Conduct**: Data collection and usage transparency

**GDPR Compliance:**
- Data minimization principles
- User privacy protection
- Informed consent for data usage
- Right to access and deletion

**Note:** Classification could change to High Risk if system evolves (Article 7).

## Technologies Used

**Machine Learning:**
- XGBoost (primary classifier)
- scikit-learn (preprocessing, evaluation)
- pandas, numpy (data manipulation)

**Data Analysis:**
- matplotlib, seaborn (visualization)
- scipy (statistical analysis)

**Deployment:**
- Streamlit (web application)
- Python 3.9+

**Database:**
- SQL for data storage and queries
- Dataset merging and joins

## Business Value

### For ANWB
- **Cost Reduction**: Fewer insurance claims due to safer driving
- **Mission Alignment**: "Prevention instead of insuring"
- **Authority Leverage**: Identify high-risk streets to advocate for municipal improvements
- **Competitive Advantage**: First street-level safety application in Netherlands

### For Drivers
- **Increased Safety**: Awareness of high-risk streets before encountering them
- **Route Flexibility**: Option to reroute around dangerous areas
- **Informed Decisions**: Data-driven understanding of local road risks
- **Lower Insurance Costs**: Safer driving behavior leads to fewer claims

### For Municipality of Breda
- **Data-Driven Infrastructure**: Identify streets requiring safety improvements
- **Resource Optimization**: Prioritize interventions based on risk levels
- **Public Safety**: Alignment with Sustainable Safety principles

## Project Impact

**Alignment with National Goals:**
- Supports Netherlands' Sustainable Safety initiative
- Follows "Mitigating" principle: reduce consequences through appropriate measures
- Contributes to reduction in road deaths (1600-1700 decrease from 1988-2007)

**ANWB Mission Alignment:**
- Insurance mission: "Preventing instead of insuring"
- Road authority status: Advocate for safer infrastructure
- Best App 2023: Enhancement to existing "Onderweg en Wegenwacht" application

## Team Contributions

**Team No. 6:**

- **Artjom Musaelans (Data Scientist)** - Model development, training, evaluation, and optimization
- **Luka Wieme** - Data Analyst (EDA and feature engineering)
- **Dominik Ptaszek** - Data Engineer #1 (data cleaning and preprocessing)
- **Lars Kerkhofs** - Data Engineer #2 (data cleaning and preprocessing)
- **Natalia Mikes** - Business Understanding (legal framework and stakeholder analysis)

## Expected Outcomes

### Technical Deliverables
- Trained XGBoost classifier for street-level risk prediction
- Comprehensive data preprocessing pipeline
- Interactive Streamlit application
- Model evaluation metrics (RMSE, Recall, MAE)

### Business Deliverables
- Street risk classification system for Breda
- Driver awareness and safety improvement tool
- Insurance cost reduction mechanism
- Municipal infrastructure improvement recommendations

## Future Enhancements

- [ ] Expand coverage beyond Breda to other municipalities
- [ ] Real-time traffic integration
- [ ] Mobile application deployment
- [ ] Integration with ANWB "Onderweg en Wegenwacht" app
- [ ] Advanced models (LSTM, GRU for time-series)
- [ ] Additional data sources (road infrastructure, lighting, vegetation)
- [ ] Severity-level predictions within risk categories
- [ ] Community-driven incident reporting

## Documentation

### Project Proposal
Comprehensive proposal document (`proposal_blockd_final.pdf`) includes:
- Introduction and background
- Problem statement with AI Business Canvas
- Detailed data description
- Complete methodology
- Legal framework and risk assessment
- Project timeline and team roles
- References and literature

### Project Structure
- **data_cleaning/**: Data preprocessing and cleaning scripts
- **eda/**: Exploratory data analysis and visualizations
- **modelling/**: Model training, evaluation, and optimization
- **streamlit_app/**: Production-ready web application
- **certificates/**: Academic completion certificates

## Responsible AI

### Transparency
- Users notified about AI-generated predictions
- Clear disclaimers about prediction limitations
- Code of conduct for data collection

### Privacy & Data Protection
- GDPR-compliant data handling
- Anonymization of personal identifiers
- Secure data storage and access controls

### Fairness
- Exclusion of sensitive demographic information
- Equal treatment across all street types
- Bias monitoring and mitigation

## Project Highlights

- **Innovative Approach**: First street-level safety prediction for Breda
- **Multi-Source Integration**: Combined 3 major datasets (ANWB + KNMI + Bron)
- **Regulatory Compliance**: EU AI Act Limited Risk + GDPR adherence
- **Production Ready**: Deployed Streamlit application
- **Team Collaboration**: 5-person team with clear role distribution
- **Real-World Impact**: Potential to reduce accidents and save lives

## Author

**Artjom Musaelans** (Student ID: 234535)  
Role: Data Scientist  
Program: Applied Data Science and Artificial Intelligence  
Institution: Breda University of Applied Sciences  
Block D Project - June 2024

## Acknowledgments

- **ANWB** for providing the Safe Driving dataset and project context
- **KNMI** (Royal Netherlands Meteorological Institute) for weather data
- **Breda University of Applied Sciences** for academic support
- **Municipality of Breda** for collaboration opportunities
- **Team No. 6** for collaborative development

---

**Project Status:** Complete  
**Deployment:** Streamlit Application  
**Date:** June 18, 2024  

*Contributing to safer roads through predictive AI and data-driven insights.*
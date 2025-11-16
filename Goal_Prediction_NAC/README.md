# Football Player Goal Prediction

A machine learning project predicting player goal-scoring performance using football statistics for NAC Breda.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## Project Overview

This project demonstrates an end-to-end data science workflow for football analytics, developed as coursework at Breda University of Applied Sciences. The analysis combines data cleaning, exploratory analysis, machine learning, and mathematical foundations to predict player goals and support strategic decisions for NAC Breda.

### Key Features

- Comprehensive data cleaning and preprocessing of 16,535+ player records
- Exploratory data analysis with 15+ visualizations
- Implementation and comparison of 6+ machine learning models
- Custom gradient descent optimization from scratch
- GDPR-compliant ethical AI framework
- RESTful API integration and database operations

## Repository Structure

```
├── Final_Deliverable.ipynb    # Complete analysis notebook
├── Final_Report.pdf           # Detailed project report
├── Certificates/              # Calculus completion certificates
└── README.md                  # This file
```

## Dataset

**16,535 players** initially → **8,923 players** after cleaning  
**116 features** including:
- Player attributes (Age, Market Value, Position, Height, Weight)
- Performance metrics (Goals, xG, Assists, Matches Played)
- Technical skills (Passing %, Duels Won %, Shots on Target %)
- Advanced stats (Progressive Runs, Defensive Actions, Save Rate %)

**Source**: NAC Breda via Breda University of Applied Sciences

## Notebook Contents

1. **Data Management**: Loading, inspection, cleaning (duplicates, missing values, outliers)
2. **Exploratory Data Analysis**: 15+ statistical analyses and visualizations
3. **Machine Learning**: Model implementation, evaluation, and optimization
4. **Database & ETL**: RESTful API integration (Giphy, Google Books)
5. **Mathematical Foundations**: Linear algebra, calculus, gradient descent implementation

## Machine Learning Results

| Model | MSE | R² Score |
|-------|-----|----------|
| **Linear Regression** | **0.0691** | **0.9903** |
| Random Forest | 0.0822 | 0.9884 |
| Gradient Boosting | 0.0049 | 0.9683 |
| Tree-Based Model | 0.0037 | 0.9756 |
| SVM | 0.1138 | 0.9843 |

**Best Model**: Linear Regression achieved **99.03% accuracy** (R²) in predicting player goals.

**Top Predictive Features** (via RFE):
1. xG (Expected Goals)
2. Goals per 90 minutes
3. Non-penalty goals
4. Non-penalty goals per 90
5. Penalties taken

## Key Findings

- **Average player age**: 25.57 years, median market value $7.5M
- **Physical correlation**: Height and weight strongly correlated (0.79)
- **Best predictor**: xG (Expected Goals) is the strongest feature for goal prediction
- **Model performance**: Linear Regression explains 99% of variance in player goals
- **Age-value relationship**: Weak negative correlation (-0.06) between age and market value

## Technologies Used

**Core**: Python, Jupyter Notebook  
**Data Analysis**: pandas, numpy, scipy, sympy  
**Machine Learning**: scikit-learn, custom gradient descent  
**Visualization**: matplotlib, seaborn  
**Other**: requests (API), mysql-connector-python

## Ethical AI

Project follows GDPR compliance and responsible AI practices:
- Data anonymization and informed consent
- Transparent model decisions and explainability
- Ethical framework based on Dignum (2019) and Floridi et al. (2018)
- Regular ethical reviews and stakeholder communication

## Documentation

- **Final_Deliverable.ipynb**: Complete technical implementation
- **Final_Report.pdf**: Comprehensive analysis including EDA, ML methodology, ethical considerations, and business recommendations
- **Certificates**: All certificates obtained during this project

## Author

**Artjoms Musaelans** (Student ID: 234535)  
Applied Data Science and Artificial Intelligence  
Breda University of Applied Sciences  
January 2024

## Acknowledgments

NAC Breda for dataset | Breda University of Applied Sciences | MSc Bram Heijligers & MA Zhanna Kozlova

---

*Developed as Block B coursework for educational purposes. Dataset anonymized.*
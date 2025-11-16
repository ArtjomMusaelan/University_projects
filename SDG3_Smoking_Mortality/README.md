# Tobacco-Related Mortality Analysis Dashboard (SDG 3)

A Power BI dashboard analyzing global tobacco-related mortality to support **UN SDG 3: Good Health & Well-Being**.

![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=powerbi&logoColor=black)
![SDG 3](https://img.shields.io/badge/SDG-3-green)

## Overview

This project investigates the relationship between tobacco use and mortality across countries and age groups from **1990 to 2019**, using official public health data.

The dashboard visualizes trends, compares countries, and highlights age-specific risks to inform public health awareness and policy decisions.

### Research Question
**Is there a significant correlation between tobacco use and mortality from tobacco-related diseases across different countries and time periods?**

## Key Insights

**Global Trend**
- Strong positive correlation between tobacco use and mortality over 30 years  
- Higher tobacco consumption → higher mortality rates

**Age Groups**

| Age Group | Correlation 1990 vs 2019 |
|------------|--------------------------|
| 15–49 yrs  | **0.99** |
| 50–69 yrs  | **0.99** |
| 70+ yrs    | **0.98** |
| Overall    | **0.84** |

**Summary**
- Mortality increased significantly across all age groups  
- Older populations show the highest absolute mortality numbers  
- Clear and consistent long-term global pattern  

## Dashboard Features

**Tabs**
1. Mortality by age group (1990–2019)
2. Country-level trends
3. Global comparison: 1990 vs 2019
4. Age-specific correlation views

**Visualizations**
- Stacked area charts (age-group mortality trends)
- Line charts (country-specific mortality progression)
- Geographic maps (death rates by country)
- Scatter plots & regression lines (1990 vs 2019 correlation)
- KPIs highlighting % change and totals

## Data Source

**Our World in Data – Smoking Dataset**  
https://ourworldindata.org/smoking

### Data Used
- Mortality from smoking by country & year (1990–2019)
- Mortality by age category (15–49, 50–69, 70+)
- Country codes & global coverage

## Methodology

- Data cleaning and transformation in **Power Query**
- DAX measures for calculations & KPIs
- Exploratory data analysis:
  - Trend analysis  
  - Correlation analysis  
  - Age-group segmentation  

## Files
```

SDG3-Smoking-Mortality/
├── SDGIndicatorsDashboard_Data_visualization_ArtjomMusaelans.pbix     # Main dashboard
├── SDGIndicatorsDashboard_Text_ArtjomMusaelans.pbix                   # Analytical report dashboard
└── Certificates/                                                      # Course certificates

```

## How to View

1. Install **Power BI Desktop**
2. Open the `.pbix` file
3. Interact with filters and charts  
   *(country selector, year range, age groups)*

## Business Relevance

This dashboard supports:

- Public health analysis  
- Evidence-based decision-making  
- Targeting high-risk demographics  
- Monitoring progress toward SDG 3 health goals  

## Future Work

- Add socioeconomic predictors (income, education, policy)
- Regression models for causal patterns
- Expand to regional health policy case studies

## Author

**Artjom Musaelans**  
Data Science & Artificial Intelligence  
Breda University of Applied Sciences

---

*Supporting UN SDG 3: Good Health and Well-Being*

# The Impact of Employee Cybersecurity Awareness on the Overall Security of Small Hotels

[![Research Status](https://img.shields.io/badge/Status-Completed-success)]()
[![Institution](https://img.shields.io/badge/Institution-BUas-blue)]()
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)]()

## Quick Links
- [📊 Data Documentation](./Data/README.md)
- [📋 Ethics & Compliance Documentation](./DMP/README.md)
- [🔬 Research Findings](#key-findings)
- [📝 Citation](#citation)

---

## Project Overview

This research project investigates how employee cybersecurity awareness impacts the overall security posture of small hotels. Conducted at Breda University of Applied Sciences in partnership with DigiWerkplaats, this study addresses the critical gap between cybersecurity knowledge and practice in small and medium-sized hospitality enterprises.

**Duration:** September 2, 2024 - November 1, 2024  
**Institution:** Breda University of Applied Sciences (BUas)  
**Partner Organization:** DigiWerkplaats Breda  

---

## Table of Contents

- [Research Question](#research-question)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Data Description](#data-description)
- [Key Recommendations](#key-recommendations)
- [Ethics & Compliance](#ethics--compliance)
- [Future Research](#future-research-directions)
- [How to Use This Repository](#how-to-use-this-repository)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Research Question

### Main Research Question
**How does employee cybersecurity awareness impact the overall security of small hotels?**

### Sub-Questions
1. Do the personal opinions of employees affect their approach to cybersecurity concerns?
2. Does the presence of an in-house IT specialist improve the cybersecurity awareness of hotel employees?
3. Does the frequency of cybersecurity training affect employee adherence to cybersecurity practices in small hotels?
4. Do the employees' ages correlate with their level of cybersecurity awareness?
5. How do small hotels educate their employees on cybersecurity practices?

---

## Key Findings

### 🎯 Training Effectiveness
- **Moderate positive correlation** (r = 0.456) between training frequency and security protocol adherence
- **Optimal training frequency:** Quarterly or biannual sessions
- Interactive, scenario-based training significantly improves engagement and retention

### 👥 Role-Specific Insights
- Employees with IT-related roles demonstrate **higher adherence levels**
- Role-specific training proves more effective than generalized approaches
- Relevance to daily tasks strongly influences practical application

### 📈 Demographic Analysis
- **No significant difference** in cybersecurity awareness across age groups
- Supports development of universal training programs
- Age and gender show minimal impact on security awareness

### 💡 Key Insight
> *"Training frequency matters more than employee demographics. Tailored, interactive training every 3-6 months yields the best results."*

---

## Methodology

### Mixed-Methods Approach

#### Quantitative Data Collection
- **Platform:** Online surveys via Qualtrics
- **Variables Measured:** Age, confidence levels, training frequency, perceived impact
- **Sample Size:** [Insert number] hotel employees
- **Data Processing:** Python with scikit-learn, Jupyter Notebooks

#### Qualitative Data Collection
- **Method:** Structured interviews (in-person or via Microsoft Teams)
- **Format:** Audio recordings (MP3) with informed consent
- **Transcription:** AI-powered with manual verification
- **Analysis:** Thematic coding and analysis

#### Analysis Tools
- Python with scikit-learn
- Jupyter Notebooks
- Statistical correlation analysis
- Thematic qualitative analysis

📘 **For detailed methodology:** See [Research Proposal](./Research_proposal.pdf)

---

## Repository Structure

```
Policy_Research_for_SMEs/
│
├── README.md                              # This file - Project overview
├── Poster.png                             # Research poster/infographic
├── Future_research.pdf                    # Identified research directions
├── Policy_paper_Cybersecurity.pdf         # Final policy recommendations
├── Research_proposal.pdf                  # Initial research design
│
├── DMP/                                   # Data Management & Ethics
│   ├── A FAIR Checklist.pdf              # FAIR data principles
│   ├── BUas Research Ethics.pdf          # Ethics review
│   ├── Codebook.md                       # Variable documentation
│   ├── Data_storage_protocol.docx        # File organization
│   ├── Informed Consent Form.docx        # Participant consent
│   ├── NWO_DMP_Data_Mangament_plan.pdf   # Data management plan
│   ├── Privacy and GDPR Checklist.pdf    # GDPR compliance 
│   └── Research Information Letter.pdf   # Study overview
│
└── Data/                                  # Research Data
    ├── README.md                          # 📊 Data documentation
    ├── Interview(audio)/                  # Interview recordings
    │   ├── Interview 1.mp3
    │   ├── Interview 2.mp3
    │   ├── Interview 3.mp3
    │   ├── Interview 4.mp3
    │   └── Interview 5.mp3
    ├── Interview(transcript)/             # Interview transcripts
    │   └── Interview_Transcript_*.pdf (5 files)
    ├── Analyzed_qualitative_data.pdf      # Thematic analysis
    ├── cleaned_survey.xlsx                # Processed survey data
    ├── Data_preparation.ipynb             # Data cleaning notebook
│   ├── EDA_with_missing_values           # Exploratory Data Analysis
│   ├── Data_visualizations.ipynb         # Visualizations from EDA
    └── FINAL-Cybersecurity-employee_*.xlsx # Raw survey responses
```

---

## Data Description

### Quantitative Data
- **Raw Survey Data:** Complete responses from Qualtrics
- **Cleaned Survey Data:** Processed and validated responses
- **Data Preparation Notebook:** Reproducible cleaning pipeline
- **Variables:** 22 quantitative variables with complete encoding

### Qualitative Data
- **Interview Recordings:** 5 audio files (MP3 format)
- **Transcripts:** 5 professionally transcribed interviews
- **Thematic Analysis:** 9 major themes identified
- **Coding:** Systematic qualitative analysis

**For complete data documentation:** See [Data README](./Data/README.md)

---

## Key Recommendations

### For Small Hotels

#### 1. Implement Regular Training
- Schedule cybersecurity training every **3-6 months**
- Balance engagement and knowledge retention
- Avoid training fatigue through optimal spacing

#### 2. Enhance Interactivity
- Incorporate **scenario-based exercises**
- Simulate real-world cybersecurity threats
- Tailor content to specific roles (front desk, night shift, management)

#### 3. Foster Security Culture
- Provide visible leadership support
- Issue regular reminders about best practices
- Integrate security naturally into daily workflows

#### 4. Universal Training Approach
- Standardize training for each role
- Ensure foundational knowledge for all employees
- Focus on inclusivity regardless of demographics

#### 5. Consider Third-Party Providers
- Explore cost-effective external training solutions
- Leverage industry best practices
- Address resource limitations

**For full recommendations:** See [Policy Paper](./Policy_paper_Cybersecurity.pdf)

---

## Ethics & Compliance

### Compliance Framework
This research adheres to strict ethical standards and data protection regulations:

✅ **BUas Research Ethics Review Board** - Approved (Medium Risk)  
✅ **GDPR Compliance** - Full compliance verified  
✅ **Netherlands Code of Conduct for Research Integrity (2018)**  
✅ **FAIR Data Principles** - Findable, Accessible, Interoperable, Reusable

### Data Protection Measures
- Encrypted storage on BUas institutional systems
- Access restricted to research team
- Two-factor authentication
- Full anonymization before public release
- 10-year retention per institutional policy

### Participant Rights
- Voluntary participation with right to withdraw
- Informed consent obtained from all participants
- Access to own data upon request
- Anonymous presentation in all publications

**For complete ethics documentation:** See [DMP README](./DMP/README.md)

---

## Future Research Directions

Based on this study's findings, we recommend future research in:

1. **Longitudinal Studies** - Track training effectiveness over 2-5 years
2. **Cost-Benefit Analysis** - Evaluate ROI of cybersecurity investments
3. **Emerging Technologies** - Assess impact of IoT devices and AI threat detection
4. **Gamification Methods** - Explore VR and mobile app training
5. **Cross-Cultural Studies** - Compare cybersecurity practices across regions
6. **Organizational Culture** - Investigate leadership and policy impact
7. **Behavioral Metrics** - Measure actual behavior beyond self-reported data

**For detailed future research suggestions:** See [Future Research](./Future_research.pdf)

---

## How to Use This Repository

### For Researchers
1. Review the [Research Proposal](./Research_proposal.pdf) for methodology
2. Check the [DMP folder](./DMP/) for ethics and data management protocols
3. Access [Data folder](./Data/) for survey and interview data
4. See [Policy Paper](./Policy_paper_Cybersecurity.pdf) for comprehensive findings

### For Practitioners
1. Review [Key Recommendations](#key-recommendations) for actionable insights
2. See [Poster.png](./Poster.png) for visual summary
3. Read [Policy Paper](./Policy_paper_Cybersecurity.pdf) for implementation guidelines

### For Students
1. Study [Research Proposal](./Research_proposal.pdf) for research design examples
2. Review [DMP folder](./DMP/) for ethics application templates
3. Examine [Data_preparation.ipynb](./Data/Data_preparation.ipynb) for data analysis techniques
4. Reference [Codebook.md](./DMP/Codebook.md) for variable documentation examples

---

## Citation

If you use this research or data, please cite:

```bibtex
@techreport{musaelans2024cybersecurity,
  title={The Impact of Employee Cybersecurity Awareness on the Overall Security of Small Hotels},
  author={Mokhonko, A. and Musaelans, A. and Wang, G. and Meijer, N. and Paskalev, P.},
  institution={Breda University of Applied Sciences},
  year={2024},
  month={October},
  address={Breda, Netherlands},
  type={Research Report}
}
```

**APA Format:**
```
Mokhonko, A., Musaelans, A., Wang, G., Meijer, N., & Paskalev, P. (2024). 
The Impact of Employee Cybersecurity Awareness on the Overall Security of Small Hotels. 
Breda University of Applied Sciences.
```

---

## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

**You are free to:**
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material

**Under the following terms:**
- Attribution — You must give appropriate credit and indicate if changes were made

---

## Contact

### Research Team
- **Institution:** Breda University of Applied Sciences
- **Partner:** DigiWerkplaats Breda
- **Email:** Mohonko.anastasia@gmail.ccom

### Data Requests
For access to anonymized research data, please contact the research team with:
- Your name and affiliation
- Intended use of the data
- Ethical approval documentation (if applicable)

---

## Acknowledgments

- **DigiWerkplaats** - Project partner and industry collaboration
- **BUas Research Ethics Review Board** - Ethics guidance and approval
- **Participating Hotels and Employees** - Research participants

---

**Last Updated:** November 10, 2025  
**Version:** 2.0  
**Status:** ✅ Completed

---

## Quick Navigation

| Section | Description | Link |
|---------|-------------|------|
| 📊 Data | Survey and interview data documentation | [Data README](./Data/README.md) |
| 📋 DMP | Ethics, GDPR, and data management | [DMP README](./DMP/README.md) |
| 📄 Policy Paper | Full recommendations and findings | [Policy Paper](./Policy_paper_Cybersecurity.pdf) |
| 🔬 Proposal | Research design and methodology | [Research Proposal](./Research_proposal.pdf) |
| 🚀 Future Research | Recommended research directions | [Future Research](./Future_research.pdf) |
| 🎨 Poster | Visual summary of findings | [Poster](./Poster.png) |

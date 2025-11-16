# Data Management Plan (DMP) Documentation

📋 **Ethics, Compliance & Data Management Documentation**

This folder contains all documentation related to data management, research ethics, and regulatory compliance for the cybersecurity awareness study.

---

## Table of Contents
- [Overview](#overview)
- [Document Summary](#document-summary)
- [Ethics Compliance](#ethics-compliance)
- [GDPR & Privacy](#gdpr--privacy)
- [FAIR Data Principles](#fair-data-principles)
- [Data Storage Protocol](#data-storage-protocol)
- [Participant Information](#participant-information)
- [Quick Reference](#quick-reference)

---

## Overview

**Purpose:** Ensure ethical, legal, and reproducible research practices  
**Compliance Standards:**
- ✅ BUas Research Ethics Guidelines
- ✅ GDPR (General Data Protection Regulation)
- ✅ Netherlands Code of Conduct for Research Integrity (2018)
- ✅ FAIR Data Principles

**Approval Status:** ✅ Approved by BUas Research Ethics Review Board  
**Risk Category:** Medium (due to personal data collection)  
**Review Date:** September 27, 2024

---

## Document Summary

| Document | Purpose | Key Information |
|----------|---------|-----------------|
| [BUas Research Ethics.pdf](#buas-research-ethics) | Ethics review approval | Risk assessment, approval status |
| [NWO_DMP_Data_Management_plan.pdf](#nwo-data-management-plan) | Complete data lifecycle | Storage, security, retention policies |
| [Privacy and GDPR Checklist.pdf](#privacy--gdpr-checklist) | Data protection compliance | Legal basis, risk assessment |
| [A FAIR Checklist.pdf](#fair-data-checklist) | Data sharing standards | Metadata, accessibility, licenses |
| [Codebook.md](#codebook) | Variable documentation | 22 quantitative + 9 qualitative variables |
| [Data_storage_protocol.docx](#data-storage-protocol) | File organization | Naming conventions, folder structure |
| [Informed Consent Form.docx](#informed-consent-form) | Participant agreement | Rights, risks, data usage |
| [Research Information Letter.pdf](#research-information-letter) | Study overview | Eligibility, procedures, contact info |

---

## Ethics Compliance

### BUas Research Ethics
**File:** `BUas Research Ethics.pdf`

#### Ethics Review Summary
- **Application Date:** September 27, 2024
- **Review Board:** BUas Research Ethics Review Board
- **Risk Category:** MEDIUM
- **Approval Status:** ✅ APPROVED

#### Risk Assessment

**Medium Risk Factors:**
- Collection of personal data (age, occupation, workplace)
- Audio recording of interviews
- Potential identification through role descriptions

**Mitigation Measures:**
- ✅ Informed consent obtained from all participants
- ✅ Data anonymization before analysis
- ✅ Secure encrypted storage
- ✅ Limited access to research team only
- ✅ Right to withdraw at any time

#### Ethics Checklist
- [x] Research purpose clearly defined
- [x] No vulnerable populations involved
- [x] No medical/clinical procedures
- [x] Risk assessment completed
- [x] Informed consent process established
- [x] Data protection measures in place
- [x] Privacy safeguards implemented
- [x] Complaint procedure provided
- [x] Supervisor approval obtained

#### Key Ethical Principles Applied

**1. Respect for Persons**
- Voluntary participation
- Informed consent
- Right to withdraw without penalty

**2. Beneficence**
- Maximize benefits to hospitality industry
- Minimize risks to participants

**3. Justice**
- Fair participant selection
- Equitable distribution of research benefits

**4. Transparency**
- Clear communication of research purpose
- Open sharing of anonymized findings

---

## GDPR & Privacy

### Privacy and GDPR Checklist
**File:** `Privacy and GDPR Checklist.pdf`

#### Legal Basis for Data Processing
**Article 6(1)(a) GDPR:** Informed consent from participants

#### Data Protection Risk Assessment

| Risk Factor | Level | Mitigation |
|-------------|-------|------------|
| Personal data collection | Medium | Informed consent + anonymization |
| Audio recordings | Medium | Secure storage + transcript anonymization |
| Workplace identification | Low | General role descriptions only |
| Data breach | Low | Encryption + access controls |

**Overall Risk Level:** 🟡 LOW TO MEDIUM

#### GDPR Compliance Measures

**Data Minimization**
- Only collect data necessary for research objectives
- No sensitive personal data (health, religion, etc.)
- Limited demographic information (age ranges only)

**Purpose Limitation**
- Data used only for stated research purposes
- No secondary use without additional consent
- Clear research objectives communicated

**Storage Limitation**
- 10-year retention period (institutional policy)
- Automatic deletion after retention period
- Regular review of stored data necessity

**Security Measures**
- ✅ Encryption at rest and in transit
- ✅ Two-factor authentication where available
- ✅ Access logs and audit trails
- ✅ Regular security updates
- ✅ Secure backup systems

**Participant Rights Under GDPR**
- ✅ Right to access their data
- ✅ Right to rectification
- ✅ Right to erasure ("right to be forgotten")
- ✅ Right to withdraw consent
- ✅ Right to data portability
- ✅ Right to object to processing

#### Data Processing Activities

**Data Controller:** Breda University of Applied Sciences  
**Data Processor:** Research team (5 students + supervisor)  
**Third-Party Processors:** None

**Processing Activities:**
1. Survey data collection (Qualtrics platform)
2. Interview recording and transcription
3. Data analysis (Python, Excel)
4. Anonymized result publication

**Data Transfers:**
- No international data transfers
- All data stored within EU (Netherlands)
- No cloud storage outside BUas systems

---

## FAIR Data Principles

### A FAIR Checklist
**File:** `A FAIR Checklist.pdf`

FAIR = **F**indable, **A**ccessible, **I**nteroperable, **R**eusable

#### Findable ✅

**Metadata Standards:**
- Dublin Core for general metadata
- DataCite for research data
- Dataset description in multiple languages

**Persistent Identifiers:**
- DOI (Digital Object Identifier) assigned to dataset
- ORCID iDs for all research team members
- Unique identifiers for all data files

**Searchable Repository:**
- Indexed in BUas institutional repository
- Keywords: cybersecurity, awareness, hotel, training, SME
- Discoverable via academic search engines

#### Accessible ✅

**Access Methods:**
- Open access to aggregated findings
- Restricted access to raw data (ethics clearance required)
- Clear access request procedure documented

**Authentication:**
- BUas SSO (Single Sign-On) for institutional access
- Secure FTP for approved external researchers
- Access log maintained for audit purposes

**Data Formats:**
- Open formats: CSV, JSON, PDF
- Proprietary formats: Excel (.xlsx) with conversion instructions
- Audio: MP3 (widely compatible)

#### Interoperable ✅

**Standards Compliance:**
- CSV files use UTF-8 encoding
- JSON follows RFC 8259 standard
- Date format: ISO 8601 (YYYY-MM-DD)

**Vocabularies:**
- Standardized terms from ISO/IEC 27001 (cybersecurity)
- Consistent variable naming across datasets
- Controlled vocabulary documented in Codebook

**Cross-Platform Compatibility:**
- Data readable in Python, R, SPSS, Excel
- No proprietary codecs or formats
- Plain text transcripts (UTF-8)

#### Reusable ✅

**Licensing:**
- Creative Commons Attribution 4.0 International (CC BY 4.0)
- Clear usage rights and restrictions
- Attribution requirements specified

**Provenance:**
- Complete documentation of data collection methods
- Processing pipeline documented in Jupyter notebooks
- Version control for all data transformations

**Quality Assurance:**
- Data validation procedures documented
- Missing data handling explained
- Inter-rater reliability reported for qualitative coding

**Documentation:**
- Comprehensive README files
- Variable codebook with full descriptions
- Methodology clearly documented

---

## Data Storage Protocol

### Data_storage_protocol.docx
**File:** `Data_storage_protocol.docx`

#### File Naming Convention

**Standard Format:**
```
<version>_<date>_<description>_<initials>.<extension>

Examples:
V1_2024-09-30_Survey_Raw_AM.xlsx
V2_2024-10-15_Interview_Transcript_P1_GW.pdf
V3_2024-11-01_Analysis_Final_NM.pdf
```

**Component Breakdown:**
- `<version>`: V1, V2, V3... (incremental)
- `<date>`: YYYY-MM-DD format
- `<description>`: Brief file description (no spaces, use underscores)
- `<initials>`: Creator initials (optional)
- `<extension>`: .xlsx, .pdf, .mp3, .ipynb, etc.

#### Folder Structure

```
Policy_Research_for_SMEs/
│
├── Data/
│   ├── Raw/                    # Original unprocessed data
│   │   ├── Survey/
│   │   └── Interviews/
│   ├── Processed/              # Cleaned and validated data
│   ├── Analysis/               # Analysis outputs
│   └── Archive/                # Outdated versions
│
├── DMP/                        # Data management documentation
│   ├── Ethics/
│   ├── Compliance/
│   └── Protocols/
│
├── Results/                    # Final outputs
│   ├── Reports/
│   ├── Visualizations/
│   └── Publications/
│
└── Documentation/              # Project documentation
    ├── Meetings/
    ├── Progress_Reports/
    └── References/
```

#### Version Control Rules

**When to Create New Version:**
- Significant data modifications
- New analysis performed
- Error corrections
- Additional data collected

**Version Numbering:**
- V1.0 → V2.0: Major changes (new variables, restructuring)
- V1.0 → V1.1: Minor changes (small corrections, formatting)

**Change Log Format:**
```
Version: V2.0
Date: 2024-10-15
Changed by: AM, GW
Changes:
  - Added 5 new survey responses
  - Corrected age variable encoding
  - Removed duplicate entry #23
```

#### Backup Protocol

**Frequency:** Daily automatic backups  
**Backup Locations:**
1. Primary: BUas institutional storage
2. Secondary: Research team OneDrive (encrypted)
3. Tertiary: External encrypted hard drive

**Retention:**
- Daily backups: 7 days
- Weekly backups: 4 weeks
- Monthly backups: 12 months
- Final version: 10+ years

---

## Participant Information

### Informed Consent Form
**File:** `Informed Consent Form.docx`

#### Consent Components

**Study Information Provided:**
- Research purpose and objectives
- Data collection procedures
- Expected time commitment (~15-45 minutes)
- Voluntary participation
- Right to withdraw at any time

**Data Usage Explained:**
- How data will be used
- Who will have access
- Anonymization procedures
- Publication plans

**Participant Rights:**
- ✅ Right to access their data
- ✅ Right to request corrections
- ✅ Right to withdraw (before analysis)
- ✅ Right to file complaints
- ✅ Confidentiality guarantees

**Consent Statement:**
> "I have read and understood the information provided. I agree to participate in this research study. I understand that my participation is voluntary and I can withdraw at any time without giving a reason."

**Signatures Required:**
- Participant signature and date
- Researcher signature and date
- Copy provided to participant

---

### Research Information Letter
**File:** `Research Information Letter.pdf`

#### Study Overview for Participants

**What is this study about?**
> This research investigates how cybersecurity training and awareness impact the security of small hotels. We want to understand what training methods work best and how to improve cybersecurity practices.

**Who can participate?**
- ✅ Age 18-65
- ✅ Currently employed in a hotel (small/medium size)
- ✅ OR hospitality management students
- ✅ Willing to complete survey or interview

**What does participation involve?**

**Option 1: Online Survey**
- Duration: 10-15 minutes
- Anonymous responses
- Questions about training and cybersecurity practices

**Option 2: Interview**
- Duration: 30-45 minutes
- In-person or via Microsoft Teams
- Audio recorded with consent
- Anonymized transcript

**Risks and Benefits**

**Potential Risks:** ⚠️ Minimal
- Time commitment
- Potential mild discomfort discussing security gaps

**Potential Benefits:** ✅
- Contribute to hospitality industry knowledge
- Improve cybersecurity practices in your workplace
- Access to study findings upon request

**Confidentiality**
- All data anonymized before analysis
- No names or hotel identifiers in publications
- Secure data storage
- Limited access to research team only

**Contact Information**
- **Research Team:** [Email]
- **Supervisor:** [Name, Email]
- **Ethics Questions:** BUas Research Ethics Board
- **Complaints:** BUas Research Integrity Officer

**Compensation:** None (voluntary participation)

---

## Codebook

### Codebook.md
**File:** `Codebook.md`

#### Variable Documentation

**Quantitative Variables (n=22)**

| Variable | Type | Scale | Values | Description |
|----------|------|-------|--------|-------------|
| `participant_id` | Nominal | N/A | P001-P999 | Unique anonymous identifier |
| `age` | Ordinal | 1-5 | 1=18-25, 2=26-35, 3=36-45, 4=46-55, 5=56-65 | Age group |
| `gender` | Nominal | 1-3 | 1=Male, 2=Female, 3=Other/Prefer not to say | Gender identity |
| `role` | Nominal | 1-6 | 1=Front desk, 2=Housekeeping, 3=Management, 4=IT, 5=F&B, 6=Other | Job role |
| `hotel_size` | Ordinal | 1-3 | 1=<50 rooms, 2=50-150 rooms, 3=>150 rooms | Hotel size category |
| `employment_duration` | Ordinal | 1-5 | 1=<1yr, 2=1-2yrs, 3=3-5yrs, 4=6-10yrs, 5=>10yrs | Length of employment |
| `training_frequency` | Ordinal | 1-5 | 1=Never, 2=Annually, 3=Biannually, 4=Quarterly, 5=Monthly | Training frequency |
| `training_format` | Nominal | 1-5 | 1=Online video, 2=In-person, 3=Workshop, 4=Self-study, 5=Mixed | Training delivery method |
| `confidence_passwords` | Ordinal | 1-5 | 1=Very low to 5=Very high | Confidence in password security |
| `confidence_phishing` | Ordinal | 1-5 | 1=Very low to 5=Very high | Confidence in identifying phishing |
| `confidence_data_protection` | Ordinal | 1-5 | 1=Very low to 5=Very high | Confidence in protecting guest data |
| `adherence_score` | Continuous | 0-100 | Calculated composite score | Overall security protocol adherence |
| `training_perceived_impact` | Ordinal | 1-5 | 1=No impact to 5=Very high impact | Perceived training effectiveness |
| `it_specialist_present` | Binary | 0-1 | 0=No, 1=Yes | In-house IT specialist availability |
| `security_incidents_year` | Discrete | 0-10+ | Number of incidents | Self-reported security incidents |

**Qualitative Themes (n=9)**

| Theme | Description | Interview Questions |
|-------|-------------|-------------------|
| Training frequency | How often cybersecurity training occurs | Q2, Q3, Q5 |
| Training content | Topics covered in training sessions | Q4, Q6 |
| Training delivery | Methods and formats used | Q7, Q8 |
| Engagement level | Participant interest and motivation | Q9, Q10 |
| Role relevance | Alignment with job responsibilities | Q11, Q12 |
| Knowledge retention | Long-term memory of training content | Q13, Q14 |
| Perceived effectiveness | Subjective assessment of training impact | Q15, Q16 |
| Self-directed learning | Independent study and research | Q17, Q18 |
| Organizational culture | Leadership and policy support | Q19, Q20 |

#### Derived Variables

**Adherence Score Calculation:**
```python
adherence_score = (
    confidence_passwords * 0.25 +
    confidence_phishing * 0.25 +
    confidence_data_protection * 0.25 +
    reported_practices * 0.25
) * 20  # Scale to 0-100
```

**Training Effectiveness Index:**
```python
effectiveness = training_frequency * training_perceived_impact / 25
# Range: 0.04 to 1.00
```

---

## Quick Reference

### Document Checklist

Before starting research:
- [x] Read Research Information Letter
- [x] Review Informed Consent Form
- [x] Sign consent form
- [x] Receive copy of signed consent

Before collecting data:
- [x] Ethics approval obtained
- [x] GDPR checklist completed
- [x] Storage protocol established
- [x] Backup systems tested

Before analyzing data:
- [x] Data anonymized
- [x] Codebook consulted
- [x] Quality checks performed
- [x] Processing documented

Before publishing:
- [x] FAIR checklist completed
- [x] Data management plan finalized
- [x] Participant rights verified
- [x] Metadata created

---

### Compliance Verification

| Standard | Status | Date Verified |
|----------|--------|---------------|
| BUas Ethics Guidelines | ✅ Compliant | 2024-09-27 |
| GDPR | ✅ Compliant | 2024-09-27 |
| Netherlands Research Integrity Code | ✅ Compliant | 2024-09-27 |
| FAIR Principles | ✅ Compliant | 2024-10-15 |

---

## Related Documentation

- 📊 [Data README](../Data/README.md)
- 📖 [Main Project README](../README.md)
- 📄 [Research Proposal](../Research_proposal.pdf)
- 📊 [Policy Paper](../Policy_paper_Cybersecurity.pdf)

---

**Last Updated:** November 10, 2025  
**Version:** 2.0  
**Maintained by:** Research Team  
**Review Date:** November 1, 2025
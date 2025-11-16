# Natural Disaster Image Classifier

An AI-powered early warning system for automatic natural disaster detection from images, designed to support emergency response agencies with rapid classification and resource mobilization.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## Project Overview

**DisasterDetect** is a deep learning application that automatically classifies natural disasters from images, enabling early warning systems and rapid emergency response. The system uses Convolutional Neural Networks (CNNs) to identify disaster types including floods, wildfires, earthquakes, tornadoes, and snowstorms.

### Key Features

- **High Accuracy Classification**: 98.3% validation accuracy, 87.5% test accuracy
- **Automated Data Collection**: Web scraping pipeline using Selenium
- **Explainable AI**: LIME and KernelSHAP implementations for model transparency
- **Responsible AI Framework**: Fairness analysis and bias mitigation strategies
- **Human-Centered Design**: A/B testing and think-aloud studies for UI/UX optimization
- **Multiple CNN Architectures**: 4 progressive iterations with performance improvements

### Application Features

- **Image Classification**: Real-time disaster type identification from photos
- **Early Warning System**: Automated alerts for detected disasters
- **Resource Mobilization**: Integration capabilities for emergency response coordination
- **Community Engagement**: User reporting and information sharing platform

## Repository Structure

```
disaster-classifier/
│
├── Classifer_of_natural_disasters.ipynb    # Complete implementation notebook
├── Image Classificator ver.B.fig           # Figma wireframe design
├── Project-Presentation.pptx               # Final presentation slides
└── README.md                               # This file
```

## Dataset

**5 Disaster Classes**: Floods, Wildfires, Earthquakes, Tornadoes, Snowstorms

- **500+ images** (100 per class initially, expanded to 300)
- **Split**: 70% training, 20% validation, 10% test
- **Source**: DuckDuckGo image search via automated Selenium scraper
- **Preprocessing**: Resizing (64x64, 128x128, 224x224), normalization, augmentation

### Data Augmentation Techniques

- Rotation (20°-45°)
- Width/Height shifting (0.2-0.3)
- Shearing and zooming (0.2-0.3)
- Brightness adjustment (0.5-1.5)
- Horizontal and vertical flipping

## Model Architecture

### Iteration Progress

**Iteration 1** - Basic CNN
- 2 Conv layers (32, 64 filters)
- Max pooling
- 128 dense units
- **Result**: 98.3% val accuracy, 87.5% test accuracy

**Iteration 2** - Dropout Regularization  
- Added dropout (0.3)
- Enhanced augmentation
- **Result**: 90.6% val accuracy, 84.4% test accuracy

**Iteration 3** - Deep Architecture
- 4 Conv layers (32, 64, 128, 256 filters)
- Dropout (0.25, 0.5)
- 512 dense units
- **Result**: 91.7% val accuracy, 90.6% test accuracy

**Iteration 4** - VGG16-style Architecture
- Batch normalization
- Advanced regularization
- **Result**: 47% val accuracy (overfitting issues)

### Best Model Performance

| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| **Accuracy** | 98.3% | 87.5% |
| **Precision** | 0.98 | 0.88 |
| **Recall** | 0.98 | 0.88 |
| **F1-Score** | 0.98 | 0.87 |

## Explainable AI Implementation

Applied multiple XAI techniques to ensure model transparency:

### Methods Used
- **LIME** (Local Interpretable Model-agnostic Explanations)
- **KernelSHAP** (Kernel SHapley Additive exPlanations)

### Segmentation Techniques Compared
- Quickshift (default)
- Felzenszwalb
- SLIC (Simple Linear Iterative Clustering)
- Watershed

### Parameters Tested
- **Number of samples**: 200, 1000, 4000, 10000, 25000
- **Reference values**: [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]
- **Interpretable models**: Ridge Regression (α=1, α=2), Linear Regression
- **Similarity kernels**: Euclidean/Cosine distance with varying kernel widths

## Responsible AI & Ethics

### Bias Identification & Mitigation

**Identified Biases**:
1. **Geographic Underrepresentation**: Urban and densely populated areas underrepresented
2. **Disaster Type Imbalance**: Overrepresentation of hurricanes/wildfires, underrepresentation of landslides/droughts

**Fairness Method Applied**: "Fairness Through Awareness"
- Added diverse geographic samples
- Balanced disaster type representation
- Increased urban disaster imagery

### Stakeholder Analysis

**Primary Stakeholder**: Emergency Response Agencies
- **Power/Influence**: High
- **Interest**: High
- **Key Needs**: Real-time alerts, system integration, accurate categorization, user-friendly interface

**Other Stakeholders**: Government authorities, NGOs, local communities, media outlets, technology providers

### DAPS Framework
- **Data-Analytic Problem**: Classify disaster type from image pixel data
- **Decision Framework**: Enable informed emergency response decisions
- **Business Value**: Reduce human/economic losses, optimize resource allocation

## Baselines & Evaluation

### Baseline Comparisons

| Model | Accuracy |
|-------|----------|
| Random Guess | 50.0% |
| Human-Level Performance | 98.3% |
| Basic MLP | 81.25% |
| **CNN (Best)** | **98.3%** |
| Custom MLP (from scratch) | 45.0% |

### Error Analysis

Conducted comprehensive error analysis including:
- Confusion matrix visualization
- Misclassified image inspection
- Pattern identification for model weaknesses
- Class-specific performance evaluation

## Human-Centered Design

### Research Methods
- **Think-Aloud Study**: User interaction analysis and feedback
- **A/B Testing**: Compared two design versions with statistical analysis (t-test)
- **Wireframe Development**: Created in Figma with iterative improvements

### Design Iterations
- Version A vs Version B comparison
- User preference analysis
- Statistical significance testing for design choices

## Technologies Used

**Deep Learning & ML**
- TensorFlow / Keras
- scikit-learn
- NumPy

**Data Collection**
- Selenium WebDriver
- Chrome WebDriver Manager

**Explainable AI**
- Xplique library
- scikit-image (segmentation)

**Visualization & Analysis**
- Matplotlib
- Seaborn
- Pillow (PIL)

**Design & Prototyping**
- Figma (wireframes)
- Microsoft Forms (surveys)

## Key Results

### Quantitative Achievements
- **98.3% validation accuracy** - Exceeds human-level performance baseline
- **87.5% test accuracy** - Strong generalization to unseen data
- **500+ images collected** - Automated scraping pipeline
- **4 model iterations** - Systematic improvement process
- **Multiple XAI methods** - Comprehensive interpretability analysis

### Qualitative Impact
- Early warning capability for emergency response
- Transparent AI decision-making through XAI
- Bias-aware model development
- Human-centered interface design
- Scalable architecture for production deployment

## Future Improvements

- [ ] Expand to 10+ disaster categories
- [ ] Real-time video stream processing
- [ ] Mobile application deployment
- [ ] Integration with satellite imagery
- [ ] Multi-language support for global deployment
- [ ] API development for emergency response systems
- [ ] Geographic information system (GIS) integration
- [ ] Severity level classification within disaster types

## Stakeholder Benefits

**Emergency Response Agencies**
- Faster disaster identification
- Automated alert systems
- Improved resource allocation

**Government Authorities**  
- Data-driven decision making
- Cost-effective monitoring
- Policy compliance support

**Local Communities**
- Timely evacuation warnings
- Safety information access
- Community engagement platform

## Documentation

- **Creative Brief**: Complete implementation in Jupyter notebook
- **Presentation**: Comprehensive project overview and results
- **Wireframe**: High-fidelity design in Figma (Version B)
- **Research Studies**: Think-aloud analysis and A/B testing results

## Author

**Artjoms Musaelans** (Student ID: 234535)  
Applied Data Science and Artificial Intelligence  
Breda University of Applied Sciences  
Block C Project

## Acknowledgments

Breda University of Applied Sciences | Project supervisors and mentors | DuckDuckGo for image search API | Emergency response community for domain insights

---

**Note**: This project was developed as part of Block C coursework focusing on Deep Learning, Responsible AI, and Human-Centered AI Design. Dataset collected through automated web scraping for educational purposes.

*Designed to save lives through early disaster detection and rapid response.*
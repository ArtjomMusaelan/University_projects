# Natural Disaster Image Classifier

An AI-powered early warning system for automatic natural disaster detection from images, designed to support emergency response agencies with rapid classification and resource mobilization.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## Project Overview

**DisasterDetect** is a deep learning application that automatically classifies natural disasters from images, enabling early warning systems and rapid emergency response. The system uses Convolutional Neural Networks (CNNs) to identify disaster types from image data.

### Key Features

- **High Accuracy Classification**: 98.3% validation accuracy, 87.5% test accuracy
- **Automated Data Collection**: Web scraping pipeline using Selenium and DuckDuckGo
- **Explainable AI**: LIME and KernelSHAP implementations for model transparency
- **Responsible AI Framework**: Fairness analysis and bias mitigation strategies
- **Human-Centered Design**: A/B testing and think-aloud studies for UI/UX optimization
- **Multiple CNN Architectures**: 4 progressive iterations with systematic evaluation

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

**2 Disaster Classes**: Floods and Wildfires

- **~300 images total** (150 per class after data cleaning)
  - Training: 208 images (70%)
  - Validation: 60 images (20%)
  - Test: 32 images (10%)
- **Source**: DuckDuckGo image search via automated Selenium scraper
- **Note**: Data collected for educational purposes only
- **Preprocessing**: Multiple resizing strategies (64×64, 128×128, 224×224), normalization, augmentation

### Data Augmentation Techniques

Progressive augmentation strategies across iterations:

| Technique | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 |
|-----------|-------------|-------------|-------------|-------------|
| Rotation | 20° | 30° | 45° | 20° |
| Width/Height Shift | 0.2 | 0.2 | 0.3 | 0.2 |
| Shear Range | 0.2 | 0.2 | 0.3 | 0.2 |
| Zoom Range | 0.2 | 0.2 | 0.3 | 0.2 |
| Brightness | - | 0.8-1.2 | 0.5-1.5 | - |
| Horizontal Flip | ✓ | ✓ | ✓ | ✓ |
| Vertical Flip | - | ✓ | ✓ | - |

## Model Architecture & Iterations

### Iteration Comparison

| Iteration | Image Size | Architecture | Dropout | Val Accuracy | Test Accuracy | Status |
|-----------|-----------|--------------|---------|--------------|---------------|--------|
| **1** | 64×64 | 2 Conv + Dense | - | **98.3%** | **87.5%** | Best |
| **2** | 224×224 | 2 Conv + Dense | 0.3 | 90.6% | 84.4% | Good |
| **3** | 128×128 | 4 Conv + Dense | 0.25, 0.5 | 91.7% | 90.6% | Good |
| **4** | 224×224 | VGG-style + BatchNorm | 0.5 | 47.0% | - | Overfitting |

### Best Model: Iteration 1

**Architecture:**
```
Input (64×64×3)
    ↓
Conv2D (32 filters, 3×3, ReLU)
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3, ReLU)
MaxPooling2D (2×2)
    ↓
Flatten
Dense (128 units, ReLU)
Dense (1 unit, Sigmoid)
```

**Training Details:**
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Early Stopping: Patience 3, monitor val_loss
- Epochs: 11 (stopped early)

### Performance Metrics - Best Model

| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| **Accuracy** | 98.3% | 87.5% |
| **Precision (Flood)** | 0.97 | 0.83 |
| **Precision (Wildfire)** | 1.00 | 0.93 |
| **Recall (Flood)** | 1.00 | 0.94 |
| **Recall (Wildfire)** | 0.97 | 0.81 |
| **F1-Score (Flood)** | 0.98 | 0.88 |
| **F1-Score (Wildfire)** | 0.98 | 0.87 |

### Why Iteration 1 Performed Best

Despite being the simplest architecture, Iteration 1 achieved superior performance due to:
- **Appropriate model complexity** for dataset size (~300 images)
- **Smaller image size (64×64)** reduced overfitting risk
- **No dropout** - the simple architecture didn't require additional regularization
- **Better generalization** - avoided learning noise from excessive augmentation

Later iterations with larger images and deeper architectures suffered from:
- Insufficient training data for complex models
- Overfitting despite regularization (especially Iteration 4)
- Increased computational requirements without performance gains

## Explainable AI Implementation

Applied multiple XAI techniques to ensure model transparency and interpretability.

### Methods Used

- **LIME** (Local Interpretable Model-agnostic Explanations)
  - Interpretable models tested: Ridge Regression (α=1, α=2), Linear Regression
  - Similarity kernels: Euclidean/Cosine distance with varying kernel widths (10.0, 45.0, 1000.0)
  - Perturbation probabilities: 0.1 to 0.9

- **KernelSHAP** (Kernel SHapley Additive exPlanations)
  - Number of samples tested: 200, 1000, 4000, 10000, 25000
  - Reference values: [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]

### Segmentation Techniques Compared

- **Quickshift** (default) - Best balance of speed and quality
- **Felzenszwalb** - Graph-based segmentation
- **SLIC** - Simple Linear Iterative Clustering
- **Watershed** - Gradient-based segmentation

### Key XAI Findings

- **Feature importance**: Model correctly focuses on fire/smoke patterns for wildfires and water/flooding patterns for floods
- **Optimal parameters**: 4000 samples with quickshift segmentation provided best explanation quality
- **Model transparency**: LIME and KernelSHAP showed consistent attribution patterns, validating model reliability

## Responsible AI & Ethics

### Bias Identification & Mitigation

**Identified Biases:**

1. **Geographic Underrepresentation**
   - Issue: Urban and densely populated areas underrepresented
   - Impact: Reduced model effectiveness in urban disaster scenarios

2. **Image Source Diversity**
   - Issue: Limited variation in environmental conditions, camera angles, and disaster contexts
   - Impact: Potential poor generalization to real-world diverse conditions

**Fairness Method Applied**: "Fairness Through Awareness"

**Mitigation Strategy:**
- Acknowledged bias limitations in dataset
- Documented need for diverse geographic samples
- Identified as key area for future improvement
- Note: Full implementation requires expanded dataset beyond project scope

### Stakeholder Analysis

**Primary Stakeholder**: Emergency Response Agencies
- **Power/Influence**: High
- **Interest**: High
- **Key Needs**: Real-time alerts, system integration, accurate categorization, user-friendly interface

**Other Stakeholders**: 
- Government authorities (High power, High interest)
- NGOs (High power, Low interest)
- Local communities (Low power, High interest)
- Media outlets (Low power, Low interest)
- Technology providers (High power, Low interest)

### DAPS Framework

- **Data-Analytic Problem**: Classify disaster type from image pixel data using pattern recognition
- **Decision Framework**: Enable informed emergency response decisions with automated disaster detection
- **Business Value**: Reduce human/economic losses, optimize resource allocation, improve community safety

## Baselines & Evaluation

### Baseline Comparisons

| Model | Accuracy | Notes |
|-------|----------|-------|
| Random Guess | 50.0% | Theoretical baseline for binary classification |
| Human-Level Performance | 98.3% | Average from 3 peer survey responses (n=20 questions) |
| Basic MLP (Keras) | 81.25% | Fully connected network with early stopping |
| **CNN Iteration 1** | **87.5%** | **Best performing model** |
| MLP From Scratch | 45.0% | Custom implementation without optimizations |

### Error Analysis

**Validation Set (Iteration 1):**
- 1 misclassification out of 60 images
- Flood misclassified as Wildfire: 1 instance
- Error rate: 1.7%

**Test Set (Iteration 1):**
- 4 misclassifications out of 32 images
- Floods misclassified as Wildfires: 1 instance
- Wildfires misclassified as Floods: 3 instances
- Error rate: 12.5%

**Common Error Patterns:**
- Ambiguous images with mixed visual features (e.g., smoke and water)
- Low-quality or distant disaster scenes
- Images with heavy atmospheric effects obscuring key features

## Human-Centered Design

### Research Methods

1. **Think-Aloud Study**
   - Methodology: User interaction observation with verbal feedback
   - Participants: 3 users
   - Key insights: Identified navigation issues and feature visibility problems

2. **A/B Testing**
   - Designs tested: Version A vs Version B
   - Metrics: Task completion time, user preference scores
   - Analysis: Independent samples t-test
   - Result: Version B selected based on statistical significance

3. **Wireframe Development**
   - Tool: Figma
   - Iterations: 2 (Version A → Version B)
   - Final design: Version B with improved layout and user flow

### Design Improvements

- Enhanced color scheme for better visibility
- Streamlined navigation structure
- Improved disaster information display
- More intuitive alert system interface

## Technologies Used

**Deep Learning & ML**
- TensorFlow 2.x / Keras
- scikit-learn
- NumPy
- Pandas

**Data Collection**
- Selenium WebDriver
- Chrome WebDriver Manager
- DuckDuckGo Search API

**Explainable AI**
- Xplique library
- scikit-image (segmentation algorithms)

**Visualization & Analysis**
- Matplotlib
- Seaborn
- Pillow (PIL)

**Design & User Research**
- Figma (wireframes)
- Microsoft Forms (surveys)
- Statistical analysis (scipy)

## Installation & Setup

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install tensorflow keras scikit-learn numpy pandas matplotlib seaborn pillow selenium xplique scikit-image

# Install WebDriver
pip install webdriver-manager

# Open Jupyter Notebook
jupyter notebook Classifer_of_natural_disasters.ipynb
```

## Usage

### Training the Model

```python
# Load and preprocess data
X_train, y_train = load_data_from_directory(train_dir)

# Build model
model = build_cnn_model(input_shape=(64, 64, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train with early stopping
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val),
                   callbacks=[EarlyStopping(patience=3)])
```

### Making Predictions

```python
# Load and preprocess image
img = load_img(image_path, target_size=(64, 64))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
disaster_type = "Wildfire" if prediction > 0.5 else "Flood"
```

## Key Results

### Quantitative Achievements

- **98.3% validation accuracy** - Exceeds human-level performance baseline
- **87.5% test accuracy** - Strong generalization to unseen data
- **~300 images collected** - Automated scraping pipeline from DuckDuckGo
- **4 model iterations** - Systematic architecture exploration
- **10+ XAI experiments** - Comprehensive interpretability analysis
- **81.25% MLP baseline** - Established performance threshold

### Qualitative Impact

- Early warning capability for emergency response
- Transparent AI decision-making through XAI
- Bias-aware model development process
- Human-centered interface design with user testing
- Scalable architecture for production deployment
- Educational demonstration of complete ML pipeline

## Lessons Learned

### Technical Insights

1. **Simplicity over complexity**: Simpler models often outperform complex ones with limited data
2. **Image size matters**: 64×64 pixels sufficient for binary classification, larger sizes risk overfitting
3. **Early stopping crucial**: Prevented overfitting across all successful iterations
4. **Data quality > quantity**: 300 well-curated images outperformed larger noisy datasets

### Process Insights

1. **Baseline importance**: MLP and human baselines provided essential performance context
2. **Iterative development**: Systematic iteration revealed that first approach was best
3. **XAI value**: Model interpretability builds trust and reveals learned patterns
4. **User research critical**: A/B testing and think-aloud studies improved design significantly

## Future Improvements

### Near-Term (3-6 months)

- [ ] Expand dataset to 1000+ images per class for better generalization
- [ ] Add 2-3 additional disaster categories (earthquakes, tornadoes)
- [ ] Implement data augmentation optimization based on XAI insights
- [ ] Deploy model as REST API for integration testing
- [ ] Add confidence thresholds for prediction reliability

### Medium-Term (6-12 months)

- [ ] Real-time video stream processing capability
- [ ] Mobile application deployment (iOS/Android)
- [ ] Multi-language support (Spanish, French, Mandarin)
- [ ] Geographic information system (GIS) integration
- [ ] Severity level classification within disaster types

### Long-Term (1-2 years)

- [ ] Integration with satellite imagery feeds
- [ ] Automated retraining pipeline with new data
- [ ] Multi-modal input (images + text + sensor data)
- [ ] Edge device deployment for offline operation
- [ ] Partnership with emergency response agencies for real-world testing

## Stakeholder Benefits

**Emergency Response Agencies**
- Faster disaster identification and classification
- Automated alert systems reduce response time
- Improved resource allocation and deployment

**Government Authorities**
- Data-driven policy and decision making
- Cost-effective disaster monitoring
- Compliance support for safety regulations

**Local Communities**
- Timely evacuation warnings and safety alerts
- Accessible safety information platform
- Community engagement and reporting features

**Technology Providers**
- Scalable AI solution for disaster management
- Integration opportunities with existing systems
- Continuous improvement through user feedback

## Documentation

- **Implementation Notebook**: Complete code with detailed explanations (`Classifer_of_natural_disasters.ipynb`)
- **Presentation Slides**: Project overview and results (`Project-Presentation.pptx`)
- **Wireframe Design**: High-fidelity UI mockup in Figma (`Image Classificator ver.B.fig`)
- **Research Studies**: Think-aloud analysis and A/B testing results (embedded in notebook)
- **DAPS Diagram**: Decision framework visualization (GitHub repository)

## Ethical Considerations

- **Data Privacy**: All images collected from public sources with no personal information
- **Bias Awareness**: Documented limitations and underrepresented populations
- **Transparency**: XAI implementation for explainable predictions
- **Educational Purpose**: Dataset collected for academic research only
- **Future Deployment**: Requires rigorous testing and validation before real-world use
- **Human Oversight**: System designed to assist, not replace, human decision-makers

## Author

**Artjoms Musaelans** (Student ID: 234535)  
Data Science and Artificial Intelligence  
Breda University of Applied Sciences  
Block C Project - Year 1

## Acknowledgments

- Breda University of Applied Sciences for project guidance
- Project supervisors and mentors for technical support
- DuckDuckGo for open image search API
- Emergency response community for domain insights
- Peer reviewers for feedback on fairness analysis and wireframe design
- Survey participants for human-level performance baseline


---

**Note**: This project was developed as part of Block C coursework focusing on Deep Learning, Responsible AI, and Human-Centered AI Design. The system is a proof-of-concept for educational purposes and requires further development, testing, and validation before deployment in real-world emergency response scenarios.

*Designed to save lives through early disaster detection and rapid response.*
# Plant Root Segmentation & Phenotyping | NPEC Hades System

An end-to-end computer vision pipeline for automated root system analysis of *Arabidopsis thaliana*, combining traditional CV methods with deep learning for high-throughput plant phenotyping applications.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## Project Overview

This project was developed for the Netherlands Plant Eco-phenotyping Centre (NPEC) to automate root segmentation and measurement for their Hades in-vitro root phenotyping system. The solution enables precise root tip localization for robotic plant inoculation and high-throughput Root System Architecture (RSA) analysis.

**Key Achievement:** Developed a complete CV pipeline that processes raw Petri dish images and outputs primary root lengths for individual plants with **82.4% F1 score** on validation data, exceeding the 80% target.

## Business Impact

- **Automation:** Replaced manual root measurement, enabling processing of 10,000+ seedlings across 2,000+ Petri dishes
- **Precision:** Achieved pixel-level accuracy for robotic liquid handling systems
- **Scalability:** Handles multiple datasets (Y2B_23, Y2B_24, Y2B_25) with varying image characteristics
- **Research Value:** Enables quantitative analysis of plant-microbe interactions and root development

## Repository Structure

```
├── image_annotation/          # Task 1: Manual labeling & peer review
├── petri_dish_detection_extraction/  # Task 2: ROI extraction (traditional CV)
├── plant_instance_segmentation/      # Task 3: Individual plant separation
├── data_preparation/          # Task 4: Dataset organization & patching
├── model_training/            # Task 5: U-Net training pipeline
├── model_training/            # Task 5.2: Model inference code 
├── individual_root_segmentation/    # Task 6: Root-level instance segmentation
├── root_system_architecture_extraction/  # Task 7: RSA extraction & measurement
├── final_pipeline/            # Task 8: Complete inference pipeline
└── models/                    # Trained segmentation models
```

## Technical Approach

### Pipeline Architecture

**Input:** Raw Petri dish images (.tif format)  
**Output:** Primary root length measurements (pixels) for each plant

The complete pipeline consists of 8 interconnected tasks:

### Task 1: Image Annotation
- **Method:** Pixel-level semantic segmentation using LabKit (ImageJ plugin)
- **Classes:** Shoot, seed, root
- **Process:** Initial labeling → peer review → correction based on feedback
- **Deliverable:** High-quality training masks with accurate shoot-root junction boundaries

### Task 2: Petri Dish Detection (Traditional CV)
- **Objective:** Extract Region of Interest (ROI) from raw images
- **Method:** Contour detection and bounding box extraction
- **Constraint:** Petri dish edge detection within ±30 pixels accuracy
- **Result:** Square cropped images focusing only on plant area

### Task 3: Plant Instance Segmentation (Traditional CV)
- **Challenge:** Separate 5 individual plants in each Petri dish
- **Method:** Traditional computer vision without deep learning
- **Application:** Enables individual plant analysis in subsequent tasks

### Task 4: Dataset Preparation
- **Organization:** Structured train/val split with proper directory hierarchy
- **Patching:** Generated 256×256 pixel patches from full images
- **Statistics:**
  - Training patches: 36,663
  - Validation patches: 5,445
- **Validation:** Random patch visualization to verify alignment

### Task 5: U-Net Model Training 
- **Architecture:** Classic U-Net with 4 encoder-decoder levels
- **Model Size:** 1,925,025 trainable parameters
- **Training Configuration:**
  - Patch size: 256×256 pixels
  - Batch size: 32
  - Optimizer: Adam
  - Loss function: Binary cross-entropy
  - Early stopping: Patience 5 epochs
- **Data Augmentation:**
  - Rotation (±20°)
  - Width/height shifts (±20%)
  - Horizontal flips
  - Shear & zoom transformations

**Performance Metrics:**
- **Validation F1 Score: 0.8238** ✓ (Target: ≥0.80)
- Validation Accuracy: 99.88%
- Best validation loss: 0.00311
- Training: 29 epochs with early stopping

### Task 6: Individual Root Segmentation
- **Input:** U-Net predicted root masks
- **Method:** Instance segmentation at root level using morphological operations
- **Output:** Separated root structures for each of the 5 plants

### Task 7: Root System Architecture Extraction
- **Technique:** Graph-based skeleton traversal using NetworkX
- **Features Extracted:**
  - Primary root identification
  - Root branch detection
  - Root tip localization
  - Primary root length measurement
- **Algorithm:** Dijkstra's shortest path with vertical orientation filtering

### Task 8: Complete Inference Pipeline
Integrated all previous tasks into a production-ready pipeline:

1. **Preprocessing:** Petri dish cropping with 20px padding
2. **Segmentation:** Patch-based U-Net inference (256×256 patches)
3. **Post-processing:** 
   - Morphological cleaning (remove small objects <300px)
   - Binary closing (7×7 kernel)
   - Gaussian blur smoothing
4. **Plant Separation:** Equal-width division into 5 plant regions
5. **RSA Extraction:** Skeletonization + graph-based primary root detection
6. **Measurement:** Euclidean distance calculation along root path

**Pipeline Output:** CSV file with primary root lengths for all plants in test dataset

## Dataset

- **Species:** *Arabidopsis thaliana*
- **Format:** Black-and-white morphometric images
- **Growth:** 5 seeds per Petri dish, photographed daily
- **Classes:** Root, shoot, seed
- **Datasets:** Y2B_23, Y2B_24, Y2B_25 (with slight variations)
- **Validation Set:** Predefined "val_..." images from Y2B_25

## Technology Stack

**Deep Learning:**
- TensorFlow/Keras 2.x
- U-Net architecture for semantic segmentation

**Computer Vision:**
- OpenCV (cv2) - Traditional CV operations
- scikit-image - Morphological processing, skeletonization
- PIL - Image I/O
- patchify - Patch generation and reconstruction

**Graph Algorithms:**
- NetworkX - Root skeleton graph traversal
- Dijkstra's algorithm for primary root path finding

**Data Processing:**
- NumPy - Numerical computations
- pandas - Results management
- Matplotlib - Visualization

## Key Features

1. **Robust Petri Dish Detection:** Works across datasets with different image characteristics (legends, varying Petri dish positions)
2. **High-Accuracy Segmentation:** 82.4% F1 score with extensive data augmentation
3. **Graph-Based Root Analysis:** Intelligent primary root detection using skeleton topology and vertical orientation filtering
4. **Production-Ready Pipeline:** Complete preprocessing → inference → post-processing workflow
5. **Scalable Processing:** Handles batches of images efficiently with patch-based approach

## Model Performance

### U-Net Training Results

| Metric | Training | Validation |
|--------|----------|------------|
| **F1 Score** | 0.6926 | **0.8238** ✓ |
| **Accuracy** | 99.68% | 99.88% |
| **Loss** | 0.0054 | 0.00311 |

- Target F1 ≥ 0.80: **ACHIEVED** ✓
- Minimum F1 ≥ 0.50: **ACHIEVED** ✓

### Training Characteristics
- Total epochs: 29 (early stopped from 30)
- Best epoch: 24
- Training time: ~3 hours on NVIDIA RTX A6000

## Technical Highlights

### Challenges Solved

1. **Dataset Variability:** Handled differences across Y2B_23/24/25 (legends, Petri dish variations)
2. **Crisscrossing Roots:** Separated overlapping root structures using instance segmentation
3. **Primary Root Detection:** Distinguished primary from lateral roots using graph topology analysis
4. **Patch-Based Inference:** Managed memory constraints with 256×256 patching strategy

### Novel Approaches

- **Vertical Ratio Filtering:** Used dy/dx ratio ≥ 1.5 to identify primary (vertical) roots
- **Multi-Start Graph Search:** Evaluated multiple potential root start points to find optimal primary root path
- **Morphological Cleanup:** Combined small object removal, binary closing, and Gaussian blur for robust predictions

## Skills Demonstrated

- Deep Learning model development and optimization
- Traditional Computer Vision techniques
- Graph algorithms and data structures
- Image preprocessing and augmentation
- End-to-end ML pipeline development
- Scientific computing and biological data analysis

## Academic Context

**Institution:** Breda University of Applied Sciences  
**Program:** Data Science & AI, Year 2  
**Client:** Netherlands Plant Eco-phenotyping Centre (NPEC)  


**Note:** This project demonstrates practical application of computer vision and deep learning to real-world agricultural research challenges, supporting NPEC's mission to harness plant potential for sustainable food production.
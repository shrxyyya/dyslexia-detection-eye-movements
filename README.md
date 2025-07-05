# Dyslexia Detection Using Eye-Movement Tracking

This project focuses on detecting dyslexia by analyzing eye-movement patterns during reading tasks. The approach leverages eye-tracking data and machine learning techniques to classify subjects as dyslexic or non-dyslexic.

---

## Dataset

- **Source:** [ETDD70: Eye-tracking Dyslexia Dataset](https://zenodo.org/records/13332134)
- **Original Size:** 70 subjects (35 dyslexic, 35 non-dyslexic)
- **Synthetic Augmentation:** Generated 140 additional synthetic subjects (2 per real subject)
- **Final Dataset Size:** 210 subjects
  - Dyslexic: 105
  - Non-dyslexic: 105
- **Features:** 23 eye-tracking features extracted from fixation data

---

## Synthetic Data Generation

### Overview
To address the limited dataset size, we implemented a sophisticated synthetic data generation pipeline that creates realistic eye-tracking data while preserving the essential characteristics of dyslexia-related eye-movement patterns.

### Generation Methodology

#### Core Algorithm
The synthetic data generation uses a perturbation-based approach implemented in `data-generator.py`:

1. **Spatial Perturbation**: 
   - Perturbs `fix_x` (horizontal fixation positions) with controlled noise
   - Preserves `fix_y` (vertical positions) unchanged to maintain reading line alignment
   - Uses adaptive perturbation factors (3-5% of spatial standard deviation)

2. **Temporal Perturbation**:
   - Applies conservative duration modifications (±5% variation)
   - Maintains original temporal structure and sequence timing
   - Preserves realistic inter-fixation intervals (saccade times)

3. **Subject ID Mapping**:
   - Original subjects: 1000-1999 range
   - Synthetic subjects: 2000-2999 and 3000-3999 ranges
   - Maintains clear separation between real and synthetic data

#### Key Features
- **Temporal Alignment**: Preserves reading sequence and timing patterns
- **Spatial Realism**: Maintains reading line structure while adding natural variation
- **Duration Consistency**: Keeps fixation durations within realistic bounds
- **No Exact Duplicates**: Ensures synthetic data doesn't contain identical entries to real data

### Quality Assurance

#### Validation Pipeline
Multiple validation scripts ensure synthetic data quality:

1. **`compare-real-fake.py`**: 
   - Detects exact and approximate matches between real and synthetic data
   - Provides quality assessment metrics
   - Ensures sufficient diversity in synthetic samples

2. **`validate-synthetic.py`**:
   - Statistical validation using KS-tests for distribution similarity
   - Mean Absolute Error (MAE) analysis
   - Pearson correlation analysis
   - Visual distribution comparisons

#### Validation Metrics
- **KS-Test p-values**: >0.05 indicates similar distributions
- **Correlation coefficients**: Close to 1.0 for preserved relationships
- **MAE scores**: Low values indicate minimal deviation from real patterns
- **Exact match detection**: Zero exact matches ensures proper diversification

#### Quality Standards
- ✓ No exact duplicates with real data
- ✓ Preserved temporal structure and reading patterns
- ✓ Maintained statistical distributions (p > 0.05 in KS-tests)
- ✓ Realistic spatial and temporal variations
- ✓ Preserved fix_y alignment for reading line consistency

### Data Processing Pipeline

1. **Data Cleaning** (`data-update.py`):
   - Removes duplicates and NaN values
   - Expands subject labels for synthetic data
   - Creates comprehensive subject mapping

2. **Synthetic Generation** (`data-generator.py`):
   - Generates 2 synthetic subjects per real subject
   - Applies controlled perturbations
   - Maintains data integrity and realism

3. **Quality Validation** (`compare-real-fake.py`, `validate-synthetic.py`):
   - Comprehensive statistical validation
   - Visual distribution analysis
   - Duplicate detection and quality assessment

### Benefits of Synthetic Data
- **Increased Dataset Size**: 3x larger dataset (210 vs 70 subjects)
- **Balanced Classes**: Equal representation of dyslexic and non-dyslexic subjects
- **Preserved Patterns**: Maintains dyslexia-related eye-movement characteristics
- **Realistic Variation**: Natural diversity without artificial artifacts
- **Validation Framework**: Comprehensive quality assurance pipeline

---

## Methodology

- **Model:** Logistic Regression
- **Feature Extraction:** Comprehensive features from eye fixation data including fixation duration, spatial distribution, saccade lengths, reading speed, regressions, and more.
- **Regularization:** Tested multiple regularization parameters (C values) to optimize model performance.
- **Cross-Validation:** 5-fold cross-validation used to select the best regularization parameter based on AUC.

---

## Model Training and Evaluation

### Regularization Parameter Tuning (C values)

| C Value | Cross-Validation AUC (mean ± std) |
|---------|-----------------------------------|
| 0.001   | 0.9508 ± 0.0715                   |
| 0.01    | 0.9572 ± 0.0721                   |
| 0.1     | **0.9585 ± 0.0619**               |
| 1       | 0.9570 ± 0.0624                   |
| 10      | 0.9570 ± 0.0574                   |

- **Best C:** 0.1 (highest CV AUC)

### Performance Metrics

| Metric                      | Training Set | Test Set  |
|-----------------------------|--------------|-----------|
| Accuracy                    | 87.50%       | 80.95%    |
| AUC (Area Under ROC Curve)  | 0.9534       | 0.9342    |

### Classification Report (Test Set)

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Non-Dyslexic  | 0.78      | 0.86   | 0.82     | 21      |
| Dyslexic      | 0.84      | 0.76   | 0.80     | 21      |

- **Overall Accuracy:** 81%
- **Macro Average:** Precision 0.81, Recall 0.81, F1-Score 0.81
- **Weighted Average:** Precision 0.81, Recall 0.81, F1-Score 0.81

---

## Usage

1. **Data Preparation:** Load and preprocess eye-tracking fixation data.
2. **Feature Extraction:** Extract 23 relevant features from fixation data.
3. **Model Training:** Train logistic regression with cross-validation to find optimal regularization.
4. **Evaluation:** Assess model performance on held-out test data.
5. **Prediction:** Use the trained model to classify new subjects based on their eye-movement features.

---

## Project Structure

- `final-data.zip` - Zip file containing original and synthetic eye-tracking data files.
- `detection.py` - Python script implementing feature extraction, model training, evaluation, and prediction.
- `data-generator.py` - Synthetic data generation script with perturbation-based methodology.
- `data-update.py` - Data cleaning and subject label expansion script.
- `compare-real-fake.py` - Quality assurance script for detecting duplicates and validating synthetic data.
- `validate-synthetic.py` - Comprehensive statistical validation and visualization script.
- `expanded_file.csv` - Subject labels file indicating dyslexia status.
- `README.md` - This documentation file.

---

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - scipy

---

## Acknowledgments

- Dataset provided by the ETDD70 project: Eye-tracking Dyslexia Dataset (Czech participants aged 9-10 years).
- Original dataset and project details available at [Zenodo ETDD70](https://zenodo.org/records/13332134).

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

---

## Acknowledgments

- Dataset provided by the ETDD70 project: Eye-tracking Dyslexia Dataset (Czech participants aged 9-10 years).
- Original dataset and project details available at [Zenodo ETDD70](https://zenodo.org/records/13332134).

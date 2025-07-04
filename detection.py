import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DyslexiaClassifier:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.features_df = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_subject_labels(self, labels_file):
        """Load subject labels from the expanded file"""
        labels_df = pd.read_csv(labels_file)
        return dict(zip(labels_df['subject_id'], labels_df['class_id']))
    
    def extract_features_from_fixations(self, fixation_file):
        """Extract comprehensive features from fixation data"""
        try:
            df = pd.read_csv(fixation_file)
            
            # Basic fixation statistics
            features = {
                'total_fixations': len(df),
                'mean_duration': df['duration_ms'].mean(),
                'std_duration': df['duration_ms'].std(),
                'median_duration': df['duration_ms'].median(),
                'min_duration': df['duration_ms'].min(),
                'max_duration': df['duration_ms'].max(),
                'total_reading_time': df['duration_ms'].sum(),
                
                # Spatial features
                'mean_fix_x': df['fix_x'].mean(),
                'mean_fix_y': df['fix_y'].mean(),
                'std_fix_x': df['fix_x'].std(),
                'std_fix_y': df['fix_y'].std(),
                'x_range': df['fix_x'].max() - df['fix_x'].min(),
                'y_range': df['fix_y'].max() - df['fix_y'].min(),
                
                # Saccade-related features (movement between fixations)
                'mean_saccade_length': np.sqrt(np.diff(df['fix_x'])**2 + np.diff(df['fix_y'])**2).mean() if len(df) > 1 else 0,
                'std_saccade_length': np.sqrt(np.diff(df['fix_x'])**2 + np.diff(df['fix_y'])**2).std() if len(df) > 1 else 0,
                
                # Reading pattern features
                'fixations_per_line': len(df) / df['aoi_line'].nunique() if df['aoi_line'].nunique() > 0 else 0,
                'unique_lines': df['aoi_line'].nunique(),
                'regression_count': sum(1 for i in range(1, len(df)) if df.iloc[i]['fix_x'] < df.iloc[i-1]['fix_x']),
                
                # Temporal features
                'reading_speed': len(df) / (df['duration_ms'].sum() / 1000) if df['duration_ms'].sum() > 0 else 0,
                'pause_frequency': sum(1 for d in df['duration_ms'] if d > df['duration_ms'].quantile(0.75)),
                
                # Advanced features
                'duration_variability': df['duration_ms'].std() / df['duration_ms'].mean() if df['duration_ms'].mean() > 0 else 0,
                'spatial_spread': np.sqrt(df['fix_x'].var() + df['fix_y'].var()),
                'fixation_density': len(df) / (df['fix_x'].max() - df['fix_x'].min() + 1) if df['fix_x'].max() != df['fix_x'].min() else 0,
            }
            
            # Handle NaN values
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    features[key] = 0
                    
            return features
            
        except Exception as e:
            print(f"Error processing {fixation_file}: {e}")
            return None
    
    def prepare_dataset(self, labels_file):
        """Prepare the complete dataset with features and labels"""
        # Load labels
        subject_labels = self.load_subject_labels(labels_file)
        
        # Find all fixation files
        fixation_files = glob.glob(os.path.join(self.data_folder_path, "*_fixations.csv"))
        
        features_list = []
        labels_list = []
        subject_ids = []
        
        for file_path in fixation_files:
            # Extract subject ID from filename
            filename = os.path.basename(file_path)
            subject_id = int(filename.split('_')[1])  # Assuming format: Subject_XXXX_...
            
            if subject_id in subject_labels:
                features = self.extract_features_from_fixations(file_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(subject_labels[subject_id])
                    subject_ids.append(subject_id)
        
        # Create DataFrame
        self.features_df = pd.DataFrame(features_list)
        self.features_df['subject_id'] = subject_ids
        self.features_df['label'] = labels_list
        
        print(f"Dataset prepared: {len(self.features_df)} subjects")
        print(f"Features: {self.features_df.shape[1] - 2}")  # Excluding subject_id and label
        print(f"Dyslexic: {sum(self.features_df['label'])}, Non-dyslexic: {len(self.features_df) - sum(self.features_df['label'])}")
        
        return self.features_df
    
    def train_model(self, test_size=0.2, C_values=[0.001, 0.01, 0.1, 1, 10]):
        """Train logistic regression with heavy regularization"""
        if self.features_df is None:
            raise ValueError("Dataset not prepared. Call prepare_dataset() first.")
        
        # Prepare features and labels
        X = self.features_df.drop(['subject_id', 'label'], axis=1)
        y = self.features_df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Find optimal C using cross-validation
        best_score = 0
        best_C = C_values[0]
        
        print("Finding optimal regularization parameter...")
        for C in C_values:
            model = LogisticRegression(C=C, random_state=42, max_iter=1000)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            mean_score = scores.mean()
            print(f"C={C}: CV AUC = {mean_score:.4f} (+/- {scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
        
        print(f"\nBest C: {best_C} with CV AUC: {best_score:.4f}")
        
        # Train final model with best C
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=best_C, random_state=42, max_iter=1000))
        ])
        
        self.model.fit(X_train, y_train)
        
        # Store data for visualization
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.best_C = best_C
        
        return self.model
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        y_train_proba = self.model.predict_proba(self.X_train)[:, 1]
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print("=== MODEL EVALUATION ===")
        print(f"Training Accuracy: {self.model.score(self.X_train, self.y_train):.4f}")
        print(f"Test Accuracy: {self.model.score(self.X_test, self.y_test):.4f}")
        print(f"Training AUC: {roc_auc_score(self.y_train, y_train_proba):.4f}")
        print(f"Test AUC: {roc_auc_score(self.y_test, y_test_proba):.4f}")
        
        print("\n=== CLASSIFICATION REPORT (Test Set) ===")
        print(classification_report(self.y_test, y_test_pred, 
                                  target_names=['Non-Dyslexic', 'Dyslexic']))
        
        return {
            'train_acc': self.model.score(self.X_train, self.y_train),
            'test_acc': self.model.score(self.X_test, self.y_test),
            'train_auc': roc_auc_score(self.y_train, y_train_proba),
            'test_auc': roc_auc_score(self.y_test, y_test_proba)
        }
    
    def plot_overfitting_analysis(self):
        """Generate comprehensive overfitting analysis plots"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Overfitting Analysis for Dyslexia Classification', fontsize=16)
        
        # 1. Learning Curves
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, self.X_train, self.y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc'
        )
        
        axes[0, 0].plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training AUC')
        axes[0, 0].plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation AUC')
        axes[0, 0].fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                               np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
        axes[0, 0].fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                               np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
        axes[0, 0].set_xlabel('Training Set Size')
        axes[0, 0].set_ylabel('AUC Score')
        axes[0, 0].set_title('Learning Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Validation Curves (C parameter)
        C_range = np.logspace(-4, 2, 10)
        train_scores, val_scores = validation_curve(
            LogisticRegression(random_state=42, max_iter=1000), 
            StandardScaler().fit_transform(self.X_train), self.y_train,
            param_name='C', param_range=C_range, cv=5, scoring='roc_auc'
        )
        
        axes[0, 1].semilogx(C_range, np.mean(train_scores, axis=1), 'o-', label='Training AUC')
        axes[0, 1].semilogx(C_range, np.mean(val_scores, axis=1), 'o-', label='Validation AUC')
        axes[0, 1].axvline(x=self.best_C, color='red', linestyle='--', label=f'Best C={self.best_C}')
        axes[0, 1].set_xlabel('Regularization Parameter (C)')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].set_title('Validation Curves (Regularization)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. ROC Curves
        y_train_proba = self.model.predict_proba(self.X_train)[:, 1]
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        fpr_train, tpr_train, _ = roc_curve(self.y_train, y_train_proba)
        fpr_test, tpr_test, _ = roc_curve(self.y_test, y_test_proba)
        
        axes[0, 2].plot(fpr_train, tpr_train, label=f'Training AUC = {auc(fpr_train, tpr_train):.3f}')
        axes[0, 2].plot(fpr_test, tpr_test, label=f'Test AUC = {auc(fpr_test, tpr_test):.3f}')
        axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('ROC Curves')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Confusion Matrix
        y_test_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix (Test Set)')
        
        # 5. Feature Importance
        feature_names = self.X_train.columns
        coefficients = self.model.named_steps['classifier'].coef_[0]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coefficients)
        }).sort_values('importance', ascending=True).tail(10)
        
        axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[1, 1].set_yticks(range(len(feature_importance)))
        axes[1, 1].set_yticklabels(feature_importance['feature'])
        axes[1, 1].set_xlabel('Absolute Coefficient Value')
        axes[1, 1].set_title('Top 10 Feature Importances')
        
        # 6. Prediction Probability Distribution
        axes[1, 2].hist(y_train_proba[self.y_train == 0], bins=20, alpha=0.7, 
                       label='Non-Dyslexic (Train)', color='blue')
        axes[1, 2].hist(y_train_proba[self.y_train == 1], bins=20, alpha=0.7, 
                       label='Dyslexic (Train)', color='red')
        axes[1, 2].hist(y_test_proba[self.y_test == 0], bins=20, alpha=0.5, 
                       label='Non-Dyslexic (Test)', color='lightblue')
        axes[1, 2].hist(y_test_proba[self.y_test == 1], bins=20, alpha=0.5, 
                       label='Dyslexic (Test)', color='pink')
        axes[1, 2].set_xlabel('Predicted Probability')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Prediction Probability Distribution')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_subject(self, fixation_file):
        """Predict dyslexia for a new subject"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        features = self.extract_features_from_fixations(fixation_file)
        if features is None:
            return None
        
        # Convert to DataFrame with same structure as training data
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.model.predict(feature_df)[0]
        probability = self.model.predict_proba(feature_df)[0, 1]
        
        return {
            'prediction': 'Dyslexic' if prediction == 1 else 'Non-Dyslexic',
            'probability': probability,
            'confidence': max(probability, 1 - probability)
        }

# Usage Example
def main():
    # Initialize classifier
    classifier = DyslexiaClassifier(data_folder_path="data")
    
    # Prepare dataset
    dataset = classifier.prepare_dataset("expanded_file.csv")
    
    # Train model with heavy regularization
    model = classifier.train_model(test_size=0.2, C_values=[0.001, 0.01, 0.1, 1, 10])
    
    # Evaluate model
    results = classifier.evaluate_model()
    
    # Generate overfitting analysis plots
    classifier.plot_overfitting_analysis()
    
    # Example prediction for new subject
    # result = classifier.predict_new_subject("path/to/new/subject/fixation/file.csv")
    # print(f"Prediction: {result}")

if __name__ == "__main__":
    main()

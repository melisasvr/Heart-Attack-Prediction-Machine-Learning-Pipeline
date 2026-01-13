"""
Heart Attack Prediction - End-to-End Classification Pipeline
Demonstrates a realistic ML workflow from data loading to model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_curve, f1_score)
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

class HeartAttackPipeline:
    """Complete pipeline for heart attack prediction"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self, df=None):
        """Load and display basic dataset information"""
        if df is not None:
            self.df = df
        elif self.data_path:
            self.df = pd.read_csv(self.data_path)
        else:
            # Create synthetic dataset for demonstration
            print("No data provided. Creating synthetic dataset...")
            self.df = self._create_synthetic_data()
        
        print("="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        print(f"Shape: {self.df.shape}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
        
        return self.df
    
    def _create_synthetic_data(self, n_samples=1000):
        """Create synthetic heart attack dataset"""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(30, 80, n_samples),
            'sex': np.random.choice([0, 1], n_samples),
            'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),
            'resting_bp': np.random.randint(90, 200, n_samples),
            'cholesterol': np.random.randint(120, 400, n_samples),
            'fasting_blood_sugar': np.random.choice([0, 1], n_samples),
            'rest_ecg': np.random.choice([0, 1, 2], n_samples),
            'max_heart_rate': np.random.randint(70, 200, n_samples),
            'exercise_angina': np.random.choice([0, 1], n_samples),
            'oldpeak': np.random.uniform(0, 6, n_samples),
            'slope': np.random.choice([0, 1, 2], n_samples),
            'num_vessels': np.random.choice([0, 1, 2, 3], n_samples),
            'thalassemia': np.random.choice([0, 1, 2, 3], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable with some logic
        risk_score = (
            df['age'] * 0.02 +
            df['chest_pain_type'] * 10 +
            df['cholesterol'] * 0.05 +
            df['max_heart_rate'] * -0.1 +
            df['exercise_angina'] * 20 +
            df['oldpeak'] * 5 +
            df['num_vessels'] * 15
        )
        df['target'] = (risk_score > risk_score.median()).astype(int)
        
        return df
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Target distribution
        print(f"\nTarget distribution:\n{self.df['target'].value_counts()}")
        print(f"\nTarget proportion:\n{self.df['target'].value_counts(normalize=True)}")
        
        # Create figure with better spacing
        fig = plt.figure(figsize=(18, 14))
        
        # Use GridSpec for better control over spacing
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, 
                              left=0.08, right=0.95, top=0.95, bottom=0.05)
        
        # 1. Target distribution
        ax1 = fig.add_subplot(gs[0, 0])
        target_counts = self.df['target'].value_counts()
        ax1.bar(['Low Risk (0)', 'High Risk (1)'], target_counts.values, 
                color=['skyblue', 'salmon'], width=0.6)
        ax1.set_title('Target Distribution', fontsize=16, pad=20, fontweight='bold')
        ax1.set_xlabel('Heart Attack Risk Category', fontsize=13)
        ax1.set_ylabel('Number of Patients', fontsize=13)
        ax1.grid(axis='y', alpha=0.3)
        # Add count labels on bars
        for i, v in enumerate(target_counts.values):
            ax1.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Age distribution by target
        ax2 = fig.add_subplot(gs[0, 1])
        low_risk_ages = self.df[self.df['target'] == 0]['age']
        high_risk_ages = self.df[self.df['target'] == 1]['age']
        
        bp = ax2.boxplot([low_risk_ages, high_risk_ages], 
                         labels=['Low Risk (0)', 'High Risk (1)'],
                         patch_artist=True,
                         widths=0.6)
        # Color the boxes
        bp['boxes'][0].set_facecolor('skyblue')
        bp['boxes'][1].set_facecolor('salmon')
        
        ax2.set_title('Age Distribution by Risk Category', fontsize=16, pad=20, fontweight='bold')
        ax2.set_xlabel('Heart Attack Risk Category', fontsize=13)
        ax2.set_ylabel('Age (years)', fontsize=13)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Correlation heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr = self.df[numeric_cols].corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax3, center=0, 
                    cbar_kws={'shrink': 0.8}, square=True)
        ax3.set_title('Feature Correlation Heatmap', fontsize=16, pad=20, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45, labelsize=9)
        ax3.tick_params(axis='y', rotation=0, labelsize=9)
        
        # 4. Feature importance preview
        ax4 = fig.add_subplot(gs[1, 1])
        correlations = self.df[numeric_cols].corrwith(self.df['target']).abs().sort_values(ascending=False)
        top_features = correlations[1:11]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        ax4.barh(range(len(top_features)), top_features.values, color=colors)
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels(top_features.index, fontsize=11)
        ax4.set_xlabel('Absolute Correlation with Target', fontsize=13)
        ax4.set_title('Top 10 Features by Correlation', fontsize=16, pad=20, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        ax4.invert_yaxis()
        
        plt.show()
        
        return correlations
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess and split data"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Separate features and target
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        
        # Handle missing values if any
        if X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            X = X.fillna(X.median())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple classification models"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.models
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Create figure with better spacing
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                              left=0.08, right=0.95, top=0.95, bottom=0.08)
        
        # 1. Model Performance Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        metrics_df = pd.DataFrame({
            name: [res['accuracy'], res['f1_score'], res['roc_auc']] 
            for name, res in self.results.items()
        }, index=['Accuracy', 'F1 Score', 'ROC AUC'])
        
        x = np.arange(len(metrics_df.columns))
        width = 0.25
        
        for i, metric in enumerate(metrics_df.index):
            ax1.bar(x + i*width, metrics_df.loc[metric], width, 
                   label=metric, alpha=0.8)
        
        ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontsize=16, pad=20, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(metrics_df.columns, rotation=15, ha='right', fontsize=10)
        ax1.legend(loc='lower right', fontsize=11)
        ax1.set_ylim([0, 1.05])
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. ROC Curves
        ax2 = fig.add_subplot(gs[0, 1])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for (name, res), color in zip(self.results.items(), colors):
            fpr, tpr, _ = roc_curve(self.y_test, res['probabilities'])
            ax2.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", 
                    linewidth=2.5, color=color)
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        ax2.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax2.set_title('ROC Curves', fontsize=16, pad=20, fontweight='bold')
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix (best model)
        ax3 = fig.add_subplot(gs[1, 0])
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
                    cbar_kws={'shrink': 0.8}, square=True,
                    annot_kws={'size': 16, 'weight': 'bold'})
        ax3.set_title(f'Confusion Matrix - {best_model_name}', 
                     fontsize=16, pad=20, fontweight='bold')
        ax3.set_ylabel('Actual Label', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax3.set_yticklabels(['Low Risk', 'High Risk'], rotation=0, fontsize=11)
        ax3.set_xticklabels(['Low Risk', 'High Risk'], rotation=0, fontsize=11)
        
        # 4. Feature Importance (for tree-based models)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'Random Forest' in self.models:
            feature_imp = pd.Series(
                self.models['Random Forest'].feature_importances_,
                index=self.X_train.columns
            ).sort_values(ascending=True).tail(10)
            
            colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(feature_imp)))
            ax4.barh(range(len(feature_imp)), feature_imp.values, color=colors)
            ax4.set_yticks(range(len(feature_imp)))
            ax4.set_yticklabels(feature_imp.index, fontsize=11)
            ax4.set_title('Top 10 Feature Importances (Random Forest)', 
                         fontsize=16, pad=20, fontweight='bold')
            ax4.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
            ax4.invert_yaxis()
        
        plt.show()
        
        # Print detailed classification report for best model
        print(f"\nDetailed Classification Report - {best_model_name}:")
        print("="*60)
        print(classification_report(self.y_test, self.results[best_model_name]['predictions'],
                                   target_names=['Low Risk', 'High Risk']))
        
        return best_model_name
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning on selected model"""
        print("\n" + "="*60)
        print(f"HYPERPARAMETER TUNING - {model_name}")
        print("="*60)
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
        
        elif model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            base_model = LogisticRegression(max_iter=1000, random_state=42)
        
        else:
            print(f"Tuning not configured for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        print("Starting grid search...")
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate tuned model
        y_pred = grid_search.predict(self.X_test_scaled)
        y_pred_proba = grid_search.predict_proba(self.X_test_scaled)[:, 1]
        
        print(f"\nTuned model performance:")
        print(f"  Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"  F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"  ROC AUC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        return grid_search.best_estimator_
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        print("\n" + "="*60)
        print("EXECUTING FULL ML PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: EDA
        self.exploratory_analysis()
        
        # Step 3: Preprocessing
        self.preprocess_data()
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Evaluate models
        best_model = self.evaluate_models()
        
        # Step 6: Optional hyperparameter tuning
        print("\nPipeline complete! Best model:", best_model)
        
        return self.results


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = HeartAttackPipeline()
    
    # Run complete pipeline
    results = pipeline.run_full_pipeline()
    
    # Optional: Perform hyperparameter tuning on best model
    # tuned_model = pipeline.hyperparameter_tuning('Random Forest')
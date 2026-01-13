# Heart Attack Prediction-Machine Learning Pipeline

- A comprehensive end-to-end machine learning project for predicting heart attack risk using patient medical data.
- This project demonstrates a complete ML workflow from data exploration to model evaluation.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Requirements](#requirements)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete machine learning pipeline for heart attack risk prediction. It includes:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and scaling
- Training multiple classification models
- Comprehensive model evaluation
- Hyperparameter tuning capabilities

## ✨ Features

- **Multiple ML Models**: Logistic Regression, Random Forest, and Gradient Boosting
- **Comprehensive Evaluation**: Accuracy, F1-Score, ROC-AUC, Cross-Validation
- **Rich Visualizations**: 8 different plots for data analysis and model comparison
- **Automated Pipeline**: Single command to run the entire workflow
- **Hyperparameter Tuning**: GridSearchCV for model optimization
- **Synthetic Data Generation**: Built-in dataset generator for testing

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-attack-prediction.git
cd heart-attack-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

### Features
The dataset includes 13 medical features:

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numeric |
| sex | Sex (0=Female, 1=Male) | Binary |
| chest_pain_type | Type of chest pain (0-3) | Categorical |
| resting_bp | Resting blood pressure (mm Hg) | Numeric |
| cholesterol | Serum cholesterol (mg/dl) | Numeric |
| fasting_blood_sugar | Fasting blood sugar > 120 mg/dl (0/1) | Binary |
| rest_ecg | Resting ECG results (0-2) | Categorical |
| max_heart_rate | Maximum heart rate achieved | Numeric |
| exercise_angina | Exercise induced angina (0/1) | Binary |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment (0-2) | Categorical |
| num_vessels | Number of major vessels (0-3) | Numeric |
| thalassemia | Thalassemia (0-3) | Categorical |
| **target** | Heart attack risk (0=Low, 1=High) | **Binary** |

### Using Your Own Data

Place your CSV file in the project directory and update the file path:

```python
pipeline = HeartAttackPipeline(data_path='your_data.csv')
results = pipeline.run_full_pipeline()
```

## 🚀 Usage

### Basic Usage

Run the complete pipeline with synthetic data:

```bash
python main.py
```

### Using Custom Data

```python
from main import HeartAttackPipeline

# Option 1: Load from CSV file
pipeline = HeartAttackPipeline(data_path='heart_attack_data.csv')
results = pipeline.run_full_pipeline()

# Option 2: Use pandas DataFrame
import pandas as pd
df = pd.read_csv('your_data.csv')
pipeline = HeartAttackPipeline()
pipeline.load_data(df)
results = pipeline.run_full_pipeline()
```

### Step-by-Step Execution

```python
pipeline = HeartAttackPipeline(data_path='heart_attack_data.csv')

# Step 1: Load data
pipeline.load_data()

# Step 2: Exploratory analysis
pipeline.exploratory_analysis()

# Step 3: Preprocess data
pipeline.preprocess_data()

# Step 4: Train models
pipeline.train_models()

# Step 5: Evaluate models
best_model = pipeline.evaluate_models()

# Step 6: Hyperparameter tuning (optional)
tuned_model = pipeline.hyperparameter_tuning('Random Forest')
```

## 📁 Project Structure

```
heart-attack-prediction/
│
├── main.py                      # Main pipeline implementation
├── heart_attack_data.csv        # Sample dataset (optional)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── outputs/                     # Generated outputs (created at runtime)
│   ├── figures/                 # Saved visualizations
│   └── models/                  # Saved model files
│
└── notebooks/                   # Jupyter notebooks (optional)
    └── exploration.ipynb        # Data exploration notebook
```

## 📈 Model Performance

Expected performance metrics on synthetic data:

| Model | Accuracy | F1-Score | ROC-AUC | CV Score |
|-------|----------|----------|---------|----------|
| Logistic Regression | 0.9950 | 0.9950 | 0.9996 | 0.9863 ± 0.0092 |
| Random Forest | 0.9250 | 0.9268 | 0.9660 | 0.9112 ± 0.0187 |
| Gradient Boosting | 0.9450 | 0.9453 | 0.9865 | 0.9163 ± 0.0242 |

**Note**: Performance will vary based on the actual dataset used.

## 📊 Visualizations
- The pipeline generates 8 comprehensive visualizations:

### Figure 1: Exploratory Data Analysis
1. **Target Distribution** - Bar chart showing class balance
2. **Age Distribution by Target** - Boxplot comparing age across risk groups
3. **Feature Correlation Heatmap** - Correlation matrix of all features
4. **Top 10 Features by Correlation** - Most important features

### Figure 2: Model Evaluation
5. **Model Performance Comparison** - Bar chart comparing all metrics
6. **ROC Curves** - ROC curves for all models
7. **Confusion Matrix** - Detailed prediction breakdown for best model
8. **Feature Importance** - Top 10 most important features (Random Forest)

## 📦 Requirements

Create a `requirements.txt` file with:

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🔮 Future Improvements
- [ ] Add more ML algorithms (XGBoost, LightGBM, Neural Networks)
- [ ] Implement SMOTE for handling imbalanced datasets
- [ ] Add feature selection techniques (RFE, LASSO)
- [ ] Create interactive dashboard with Streamlit/Dash
- [ ] Add model deployment capabilities (Flask API)
- [ ] Implement automated feature engineering
- [ ] Add SHAP values for model interpretability
- [ ] Create Docker container for easy deployment
- [ ] Add unit tests and CI/CD pipeline
- [ ] Implement ensemble methods (voting, stacking)

## 🤝 Contributing
- Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors
- **Melisa Sever** 

## 🙏 Acknowledgments
- Dataset inspired by UCI Heart Disease Dataset
- Built with scikit-learn, pandas, and matplotlib
- Thanks to the open-source community for amazing tools


---

**⭐ If you found this project helpful, please consider giving it a star!**

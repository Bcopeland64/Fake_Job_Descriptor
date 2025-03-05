# Fake Job Posting Detector

This project provides multiple implementations of a fake job posting detector using different machine learning approaches, from lightweight traditional ML to advanced transformer-based models.

## Implementations

### 1. Light Fake Job Detector (Recommended for Efficiency)

`Light_Fake_Job_Detector.py` is a streamlined implementation that prioritizes speed and efficiency:

- Uses TF-IDF vectorization with Logistic Regression or Random Forest
- Includes feature engineering for technical and suspicious keywords
- Handles class imbalance with SMOTE
- Very fast training (seconds vs. hours for transformer models)
- Low resource requirements (runs on CPU)
- Achieves good performance (~90% accuracy)

### 2. Advanced Fake Job Detector (Best Accuracy)

`Improved_Fake_Job_Detector.py` is a comprehensive implementation focusing on maximum accuracy:

- Fine-tunes BERT/DistilBERT for text classification
- Implements threshold optimization for best F1 score
- Includes extensive feature engineering
- Provides detailed model interpretability
- Highest accuracy (~92-95%)
- Requires significant computational resources (GPU recommended)

### 3. Basic Implementation

`Fake_Job_Descriptor.py` is a simple implementation for reference:

- Basic BERT model for job posting analysis
- Limited feature engineering
- Streamlit interface for quick testing

## How to Use

### Requirements
Install the required packages:

```bash
# For the lightweight implementation only
pip install streamlit pandas numpy scikit-learn matplotlib seaborn imblearn

# For all implementations (including transformer-based)
pip install streamlit pandas numpy scikit-learn torch transformers imblearn matplotlib seaborn shap
```

### Running the App
```bash
# Recommended lightweight version
streamlit run Light_Fake_Job_Detector.py

# Advanced version (requires more resources)
streamlit run Improved_Fake_Job_Detector.py
```

Each app has two modes:
1. **Job Analysis**: Analyze job postings to determine if they're fake
2. **Model Training**: Train a new model on the fake job postings dataset

## Performance Comparison

| Implementation | Training Time | Accuracy | Resource Usage | Model Size |
|----------------|--------------|----------|---------------|------------|
| Light (Logistic) | ~30 seconds | ~88-90% | Very Low (CPU) | ~10-20MB |
| Light (Random Forest) | ~1 minute | ~89-91% | Low (CPU) | ~50-100MB |
| Advanced (DistilBERT) | ~30-60 minutes | ~91-93% | High (GPU recommended) | ~300MB |
| Advanced (BERT) | ~1-2 hours | ~92-95% | Very High (GPU required) | ~400MB+ |

## Choosing the Right Implementation

- **Limited resources or quick results**: Use `Light_Fake_Job_Detector.py` with Logistic Regression
- **Balance of accuracy and speed**: Use `Light_Fake_Job_Detector.py` with Random Forest
- **Maximum accuracy for critical applications**: Use `Improved_Fake_Job_Detector.py` with DistilBERT
- **Best possible accuracy regardless of resources**: Use `Improved_Fake_Job_Detector.py` with BERT

## Files
- `Light_Fake_Job_Detector.py`: Lightweight implementation (recommended)
- `Improved_Fake_Job_Detector.py`: Advanced implementation
- `Fake_Job_Descriptor.py`: Basic implementation
- `fake_job_postings.csv`: Dataset containing genuine and fraudulent job postings
- `job_train.csv`: Alternative training dataset
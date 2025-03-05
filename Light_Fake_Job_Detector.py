import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Function to clean text
def text_cleaning(text):
    if not isinstance(text, str):
        return ""
    # More advanced text cleaning
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Enhanced list of technical indicators for feature engineering
tech_indicators = [
    'python', 'java', 'javascript', 'aws', 'azure', 'docker', 'kubernetes', 'react', 'angular', 'vue', 'node.js', 'sql',
    'nosql', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'machine learning', 'deep learning', 'api', 'rest',
    'graphql', 'devops', 'agile', 'scrum', 'blockchain', 'cybersecurity', 'iot', 'big data', 'hadoop', 'spark', 'git',
    'jenkins', 'linux', 'bash', 'C++', 'JavaScript', 'HTML', 'CSS', 'GitHub', 'CI/CD', 'Cloud Computing', 'Data Science',
    'Artificial Intelligence', 'Natural Language Processing', 'Computer Vision', 'Network Security', 'Cloud Storage',
    'Serverless Computing', 'Microservices', 'DevOps Automation', 'Continuous Integration', 'Continuous Deployment',
    'Containerization', 'Virtualization', 'typescript', 'php', 'ruby', 'django', 'flask', 'spring', 'bootstrap',
    'tailwind', 'mongodb', 'postgresql', 'mysql', 'oracle', 'redis', 'elasticsearch', 'selenium', 'pytest', 'jest',
    'mocha', 'jira', 'trello', 'figma', 'sketch', 'photoshop', 'illustrator', 'ux/ui', 'responsive design',
    'mobile development', 'ios', 'android', 'swift', 'kotlin', 'flutter', 'react native'
]

# List of suspicious indicators that might suggest a fake job posting
suspicious_indicators = [
    'urgent', 'immediate start', 'work from home', 'no experience', 'high salary', 'easy money',
    'quick money', 'earn money', 'part time', 'flexible hours', 'flexible schedule',
    'be your own boss', 'work anywhere', 'unlimited earning', 'income potential',
    'financial freedom', 'get rich', 'minimal work', 'opportunity', 'free training',
    'paid training', 'no investment', 'investment required', 'free registration',
    'registration fee', 'full refund', 'money back', 'guarantee', 'limited spots',
    'act now', 'apply now', 'don\'t wait', 'don\'t miss', 'hurry', 'best opportunity',
    'life changing', 'dream job', 'exclusive', 'secret', 'confidential', 'hidden',
    'private', 'restricted', 'unique', 'special', 'rare'
]

class LightFakeJobDetector:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.threshold = 0.5  # Default threshold, will be optimized later
        self.initialize_model()
        
    def initialize_model(self):
        if self.model_type == 'logistic':
            self.model = LogisticRegression(C=1.0, max_iter=200, class_weight='balanced')
        else:  # Use RandomForest by default
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')
    
    def preprocess_data(self, df):
        """Preprocess the dataframe for training"""
        print("Preprocessing data...")
        # Clean text columns
        for col in ['title', 'description', 'requirements']:
            if col in df.columns:
                df[col] = df[col].apply(text_cleaning)
        
        # Combine text features for better context
        df['combined_text'] = ''
        for col in ['title', 'description', 'requirements']:
            if col in df.columns:
                df['combined_text'] += ' ' + df[col].fillna('')
        
        # Feature engineering - count technical keywords
        df['tech_keyword_count'] = df['combined_text'].apply(
            lambda x: sum(1 for word in tech_indicators if word in x)
        )
        
        # Feature engineering - count suspicious keywords
        df['suspicious_keyword_count'] = df['combined_text'].apply(
            lambda x: sum(1 for word in suspicious_indicators if word in x)
        )
        
        # Add text length as a feature
        df['text_length'] = df['combined_text'].apply(len)
        
        return df
    
    def extract_features(self, df):
        """Extract features from preprocessed data"""
        X_text = self.vectorizer.fit_transform(df['combined_text'])
        
        # Add engineered features
        X_features = df[['tech_keyword_count', 'suspicious_keyword_count', 'text_length']].values
        
        # Convert sparse matrix to numpy array and concatenate with other features
        X_text_dense = X_text.toarray()
        X = np.hstack((X_text_dense, X_features))
        
        return X
    
    def handle_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def train(self, train_df, val_df=None):
        """Train the model"""
        if val_df is None:
            # Split into train (80%) and validation (20%)
            train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['fraudulent'])
        
        print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
        
        # Extract features
        X_train = self.extract_features(train_df)
        y_train = train_df['fraudulent'].values
        
        X_val = self.vectorizer.transform(val_df['combined_text']).toarray()
        X_val_features = val_df[['tech_keyword_count', 'suspicious_keyword_count', 'text_length']].values
        X_val = np.hstack((X_val, X_val_features))
        y_val = val_df['fraudulent'].values
        
        # Handle class imbalance
        X_train_resampled, y_train_resampled = self.handle_imbalance(X_train, y_train)
        
        # Train the model
        print("Training model...")
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        
        # Optimize threshold if using logistic regression
        if self.model_type == 'logistic':
            self.optimize_threshold(X_val, y_val)
        
        # Return validation metrics
        return {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1
        }
    
    def optimize_threshold(self, X_val, y_val):
        """Find the optimal classification threshold (for logistic regression)"""
        if self.model_type != 'logistic':
            return
        
        # Get probabilities
        y_val_probs = self.model.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            y_val_pred = (y_val_probs >= threshold).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            f1_scores.append(f1)
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        self.threshold = thresholds[best_idx]
        
        print(f"Optimal threshold: {self.threshold:.2f} with F1 Score: {f1_scores[best_idx]:.4f}")
    
    def predict(self, text):
        """Predict if a job posting is fake"""
        # Preprocess text
        cleaned_text = text_cleaning(text)
        
        # Extract features
        X_text = self.vectorizer.transform([cleaned_text]).toarray()
        
        # Extract additional features
        tech_count = sum(1 for word in tech_indicators if word in cleaned_text)
        suspicious_count = sum(1 for word in suspicious_indicators if word in cleaned_text)
        text_length = len(cleaned_text)
        
        X_features = np.array([[tech_count, suspicious_count, text_length]])
        X = np.hstack((X_text, X_features))
        
        # Get prediction
        if self.model_type == 'logistic':
            # For logistic regression, we can get probabilities
            probs = self.model.predict_proba(X)[0]
            genuine_prob = probs[0] * 100
            fake_prob = probs[1] * 100
            is_fake = fake_prob/100 >= self.threshold
        else:
            # For random forest, we can still get probabilities
            probs = self.model.predict_proba(X)[0]
            genuine_prob = probs[0] * 100
            fake_prob = probs[1] * 100
            is_fake = fake_prob/100 >= 0.5  # Use default threshold for RF
        
        # Get detected indicators
        tech_indicators_present = [word for word in tech_indicators if word in cleaned_text]
        suspicious_indicators_present = [word for word in suspicious_indicators if word in cleaned_text]
        
        return {
            'is_fake': is_fake,
            'genuine_prob': genuine_prob,
            'fake_prob': fake_prob,
            'tech_indicators': tech_indicators_present,
            'suspicious_indicators': suspicious_indicators_present,
            'tech_count': tech_count,
            'suspicious_count': suspicious_count
        }
    
    def save_model(self, path):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'threshold': self.threshold,
            'model_type': self.model_type
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path):
        """Load model from file"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.threshold = model_data['threshold']
        self.model_type = model_data['model_type']

def train_model():
    """Function to train the model using the available data"""
    # Load data
    try:
        df = pd.read_csv('fake_job_postings.csv')
    except:
        df = pd.read_csv('job_train.csv')
    
    print(f"Loaded dataset with {len(df)} rows and {df.columns.shape[0]} columns")
    
    # Check for class imbalance
    print("\nClass distribution:")
    print(df['fraudulent'].value_counts())
    
    # Preprocess data
    detector = LightFakeJobDetector(model_type='logistic')
    df = detector.preprocess_data(df)
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fraudulent'])
    
    # Train the model
    metrics = detector.train(train_df)
    
    # Save the model
    detector.save_model('./light_fake_job_detector_model.pkl')
    
    # Evaluate on test set
    evaluate_model(detector, test_df)
    
    return detector, metrics

def evaluate_model(detector, test_df):
    """Evaluate the model on test data"""
    print("\nEvaluating model on test data...")
    
    # Extract features from test data
    X_test = detector.vectorizer.transform(test_df['combined_text']).toarray()
    X_test_features = test_df[['tech_keyword_count', 'suspicious_keyword_count', 'text_length']].values
    X_test = np.hstack((X_test, X_test_features))
    y_test = test_df['fraudulent'].values
    
    # Get predictions
    if detector.model_type == 'logistic':
        y_probs = detector.model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= detector.threshold).astype(int)
    else:
        y_pred = detector.model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    try:
        from sklearn.metrics import roc_curve, auc
        if detector.model_type == 'logistic':
            y_probs = detector.model.predict_proba(X_test)[:, 1]
        else:
            y_probs = detector.model.predict_proba(X_test)[:, 1]
            
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Fake Job Detection')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')
        plt.close()
        
        print("\nROC curve saved as 'roc_curve.png'")
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

# Streamlit UI
def run_streamlit_app():
    st.set_page_config(page_title="Light Fake Job Detector", layout="wide")
    
    st.title("Light Fake Job Posting Detector")
    st.write("A faster, more resource-efficient approach using traditional ML")
    
    # Sidebar for app navigation
    app_mode = st.sidebar.selectbox(
        "Select App Mode",
        ["Job Analysis", "Model Training"]
    )
    
    # Try to load the pre-trained model
    model_path = './light_fake_job_detector_model.pkl'
    detector = LightFakeJobDetector()
    model_loaded = False
    
    try:
        detector.load_model(model_path)
        model_loaded = True
        st.sidebar.success("✅ Model loaded successfully!")
    except:
        st.sidebar.warning("⚠️ Pre-trained model not found. You can train a new model in the 'Model Training' section.")
    
    if app_mode == "Job Analysis":
        analyze_job_posting(detector, model_loaded)
    else:
        train_new_model()

def analyze_job_posting(detector, model_loaded):
    st.header("Job Posting Analysis")
    
    # Text input for job description
    job_text = st.text_area(
        "Enter the job posting text to analyze:",
        height=300
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        analyze_button = st.button("Analyze Job Posting")
    
    with col2:
        clear_button = st.button("Clear")
    
    if clear_button:
        st.experimental_rerun()
    
    if analyze_button and job_text:
        if not model_loaded:
            st.error("⚠️ No model loaded. Please train a model first.")
            return
        
        with st.spinner("Analyzing job posting..."):
            result = detector.predict(job_text)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            # Display overall assessment
            if result['is_fake']:
                st.error(f"⚠️ This job posting is likely FAKE ({result['fake_prob']:.1f}% certainty)")
            else:
                st.success(f"✅ This job posting is likely GENUINE ({result['genuine_prob']:.1f}% certainty)")
            
            # Display probability meters
            st.write("### Probability Assessment")
            st.write("Genuine:")
            st.progress(result['genuine_prob']/100)
            st.write(f"{result['genuine_prob']:.1f}%")
            
            st.write("Fake:")
            st.progress(result['fake_prob']/100)
            st.write(f"{result['fake_prob']:.1f}%")
        
        with col2:
            # Display supporting evidence
            st.write("### Supporting Evidence")
            
            # Technical indicators
            st.write(f"**Technical terms detected:** {result['tech_count']}")
            if result['tech_indicators']:
                st.write(", ".join(result['tech_indicators'][:10]))
                if len(result['tech_indicators']) > 10:
                    st.write(f"...and {len(result['tech_indicators']) - 10} more")
            
            # Suspicious indicators
            st.write(f"**Suspicious terms detected:** {result['suspicious_count']}")
            if result['suspicious_indicators']:
                st.write(", ".join(result['suspicious_indicators'][:10]))
                if len(result['suspicious_indicators']) > 10:
                    st.write(f"...and {len(result['suspicious_indicators']) - 10} more")
        
        # Detailed analysis
        st.write("### Detailed Analysis")
        
        if result['is_fake']:
            st.write("**Why this might be a fake job posting:**")
            reasons = [
                "Uses vague language without specific job requirements" if result['tech_count'] < 3 else None,
                "Contains suspicious words commonly found in fraudulent postings" if result['suspicious_count'] > 2 else None,
                "Lacks specific technical requirements for a technical position" if "tech" in job_text.lower() and result['tech_count'] < 5 else None,
                "Promises high rewards with minimal effort" if any(word in job_text.lower() for word in ['easy money', 'quick money', 'high salary']) else None,
                "Requires upfront payment or investment" if any(word in job_text.lower() for word in ['payment', 'invest', 'fee', 'cost']) else None
            ]
            
            for reason in reasons:
                if reason:
                    st.write(f"- {reason}")
            
            st.write("")
            st.write("**Safety tips:**")
            st.write("- Never pay money to apply for a job")
            st.write("- Research the company thoroughly")
            st.write("- Check if the company has a professional website")
            st.write("- Verify the company's contact information")
            st.write("- Be cautious of jobs with extremely high salaries for minimal qualifications")
        else:
            st.write("**Why this looks like a legitimate job posting:**")
            reasons = [
                "Contains specific technical requirements" if result['tech_count'] >= 5 else None,
                "Uses precise language for job responsibilities" if len(job_text) > 500 else None,
                "Minimal use of suspicious terms" if result['suspicious_count'] < 3 else None,
                "Includes detailed requirements and qualifications" if "requirements" in job_text.lower() or "qualifications" in job_text.lower() else None
            ]
            
            for reason in reasons:
                if reason:
                    st.write(f"- {reason}")
            
            st.write("")
            st.write("**Note:** Even with genuine job postings, always verify the company and position before sharing personal information.")

def train_new_model():
    st.header("Model Training")
    
    st.write("""
    This section allows you to train a new model on the fake job postings dataset.
    Training a new model will be much faster than the transformer-based approach.
    """)
    
    model_type = st.selectbox(
        "Select model type",
        ["logistic", "random_forest"],
        help="Logistic Regression is faster, Random Forest might be slightly more accurate."
    )
    
    train_button = st.button("Train Model")
    
    if train_button:
        with st.spinner("Training model... This should be quick."):
            # Create progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Set up model
            detector = LightFakeJobDetector(model_type=model_type)
            
            # Load data
            try:
                df = pd.read_csv('fake_job_postings.csv')
                progress_text.text("Loaded fake_job_postings.csv")
            except:
                try:
                    df = pd.read_csv('job_train.csv')
                    progress_text.text("Loaded job_train.csv")
                except:
                    st.error("Could not find dataset files. Make sure fake_job_postings.csv or job_train.csv exists in the current directory.")
                    return
            
            progress_bar.progress(20)
            progress_text.text("Preprocessing data...")
            
            # Preprocess data
            df = detector.preprocess_data(df)
            
            progress_bar.progress(40)
            progress_text.text("Splitting dataset into train and test sets...")
            
            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fraudulent'])
            
            progress_bar.progress(50)
            progress_text.text("Training model...")
            
            # Train model
            detector.train(train_df)
            
            progress_bar.progress(80)
            progress_text.text("Evaluating model...")
            
            # Evaluate model
            evaluate_model(detector, test_df)
            
            progress_bar.progress(90)
            progress_text.text("Saving model...")
            
            # Save model
            detector.save_model('./light_fake_job_detector_model.pkl')
            
            progress_bar.progress(100)
            progress_text.text("Training complete!")
        
        st.success("Model training completed successfully!")
        st.info("Switch to the 'Job Analysis' tab to analyze job postings with your new model.")

if __name__ == "__main__":
    run_streamlit_app()
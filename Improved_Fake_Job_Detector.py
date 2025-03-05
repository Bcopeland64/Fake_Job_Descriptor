import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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

class FakeJobDetector:
    def __init__(self, model_type='distilbert'):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.5  # Default threshold, will be optimized later
        self.initialize_model()
        
    def initialize_model(self):
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        else:  # Use DistilBERT by default (faster, almost as accurate)
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        
        self.model.to(self.device)
    
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
    
    def prepare_bert_features(self, texts, max_length=256):
        """Convert text to BERT features"""
        encodings = self.tokenizer(texts.tolist(), 
                                  truncation=True, 
                                  padding='max_length', 
                                  max_length=max_length, 
                                  return_tensors='pt')
        
        return encodings
    
    def handle_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def train(self, train_df, val_df=None, epochs=3, batch_size=16, learning_rate=2e-5):
        """Train the model with fine-tuning"""
        if val_df is None:
            # Split into train (80%) and validation (20%)
            train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['fraudulent'])
        
        print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
        
        # Prepare data
        train_texts = train_df['combined_text']
        train_labels = train_df['fraudulent'].values
        
        val_texts = val_df['combined_text']
        val_labels = val_df['fraudulent'].values
        
        # Tokenize data
        train_encodings = self.prepare_bert_features(train_texts)
        val_encodings = self.prepare_bert_features(val_texts)
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'], 
            train_encodings['attention_mask'],
            torch.tensor(train_labels)
        )
        
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(val_labels)
        )
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size
        )
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Total number of training steps
        total_steps = len(train_dataloader) * epochs
        
        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        
        # Store metrics for plotting
        training_stats = []
        
        # For each epoch
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            
            # Reset metrics
            total_train_loss = 0
            total_train_accuracy = 0
            
            # Training
            for batch in train_dataloader:
                # Unpack the inputs from our dataloader
                b_input_ids = batch[0].to(self.device)
                b_attention_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                
                # Clear gradients
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1).flatten()
                accuracy = (preds == b_labels).cpu().numpy().mean()
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Update tracking variables
                total_train_loss += loss.item()
                total_train_accuracy += accuracy
            
            # Calculate average loss and accuracy over the training data
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_train_accuracy = total_train_accuracy / len(train_dataloader)
            
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Training Accuracy: {avg_train_accuracy:.4f}")
            
            # Validation
            self.model.eval()
            
            total_val_loss = 0
            total_val_accuracy = 0
            all_preds = []
            all_labels = []
            
            # Evaluate data for one epoch
            for batch in val_dataloader:
                # Unpack the inputs from our dataloader
                b_input_ids = batch[0].to(self.device)
                b_attention_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                
                # Don't compute gradients
                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(
                        input_ids=b_input_ids,
                        attention_mask=b_attention_mask,
                        labels=b_labels
                    )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1).flatten()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(b_labels.cpu().numpy())
                accuracy = (preds == b_labels).cpu().numpy().mean()
                
                # Update tracking variables
                total_val_loss += loss.item()
                total_val_accuracy += accuracy
            
            # Calculate average loss and accuracy over validation data
            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_accuracy = total_val_accuracy / len(val_dataloader)
            
            # Calculate other metrics
            val_precision = precision_score(all_labels, all_preds)
            val_recall = recall_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds)
            
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            print(f"  Validation Accuracy: {avg_val_accuracy:.4f}")
            print(f"  Validation Precision: {val_precision:.4f}")
            print(f"  Validation Recall: {val_recall:.4f}")
            print(f"  Validation F1: {val_f1:.4f}")
            
            # Store stats for visualization
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': avg_train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            })
            
            # Set back to training mode
            self.model.train()
        
        # Find the best threshold using validation data
        self.optimize_threshold(val_dataloader)
        
        # Return training stats for visualization
        return training_stats
    
    def optimize_threshold(self, val_dataloader):
        """Find the optimal classification threshold"""
        self.model.eval()
        
        all_probs = []
        all_labels = []
        
        # Collect all predictions and labels
        for batch in val_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_attention_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask
                )
            
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
        
        # Try different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            preds = [1 if prob >= threshold else 0 for prob in all_probs]
            f1 = f1_score(all_labels, preds)
            f1_scores.append(f1)
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        self.threshold = thresholds[best_idx]
        
        print(f"Optimal threshold: {self.threshold:.2f} with F1 Score: {f1_scores[best_idx]:.4f}")
    
    def predict(self, text):
        """Predict if a job posting is fake"""
        self.model.eval()
        
        # Preprocess text
        cleaned_text = text_cleaning(text)
        
        # Extract additional features
        tech_count = sum(1 for word in tech_indicators if word in cleaned_text)
        suspicious_count = sum(1 for word in suspicious_indicators if word in cleaned_text)
        
        # Create encodings
        encodings = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        genuine_prob = probs[0, 0].item() * 100
        fake_prob = probs[0, 1].item() * 100
        
        # Get detected indicators
        tech_indicators_present = [word for word in tech_indicators if word in cleaned_text]
        suspicious_indicators_present = [word for word in suspicious_indicators if word in cleaned_text]
        
        # Determine prediction using optimized threshold
        is_fake = fake_prob/100 >= self.threshold
        
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
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save threshold and other config
        config = {
            'threshold': self.threshold,
            'model_type': self.model_type
        }
        
        with open(f"{path}/config.pkl", 'wb') as f:
            pickle.dump(config, f)
    
    def load_model(self, path):
        """Load model from file"""
        if self.model_type == 'bert':
            self.model = BertForSequenceClassification.from_pretrained(path)
            self.tokenizer = BertTokenizer.from_pretrained(path)
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        
        self.model.to(self.device)
        
        # Load threshold and other config
        try:
            with open(f"{path}/config.pkl", 'rb') as f:
                config = pickle.load(f)
                self.threshold = config['threshold']
        except:
            print("Config file not found, using default threshold.")

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
    detector = FakeJobDetector(model_type='distilbert')
    df = detector.preprocess_data(df)
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fraudulent'])
    
    # Train the model
    training_stats = detector.train(train_df, epochs=3, batch_size=16)
    
    # Save the model
    detector.save_model('./fake_job_detector_model')
    
    # Evaluate on test set
    evaluate_model(detector, test_df)
    
    return detector, training_stats

def evaluate_model(detector, test_df):
    """Evaluate the model on test data"""
    print("\nEvaluating model on test data...")
    
    # Prepare test data
    test_texts = test_df['combined_text']
    test_labels = test_df['fraudulent'].values
    
    # Tokenize data
    test_encodings = detector.prepare_bert_features(test_texts)
    
    # Create PyTorch dataset
    test_dataset = TensorDataset(
        test_encodings['input_ids'],
        test_encodings['attention_mask'],
        torch.tensor(test_labels)
    )
    
    # Create DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16
    )
    
    # Evaluation
    detector.model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in test_dataloader:
        b_input_ids = batch[0].to(detector.device)
        b_attention_mask = batch[1].to(detector.device)
        b_labels = batch[2].to(detector.device)
        
        with torch.no_grad():
            outputs = detector.model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask
            )
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())
    
    # Apply threshold
    all_preds = [1 if prob >= detector.threshold else 0 for prob in all_probs]
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Plot ROC curve
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
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
    st.set_page_config(page_title="Advanced Fake Job Detector", layout="wide")
    
    st.title("Advanced Fake Job Posting Detector")
    
    # Sidebar for app navigation
    app_mode = st.sidebar.selectbox(
        "Select App Mode",
        ["Job Analysis", "Model Training"]
    )
    
    # Try to load the pre-trained model
    model_path = './fake_job_detector_model'
    detector = FakeJobDetector()
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
    Training a new model will take some time to complete.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select model type",
            ["distilbert", "bert"],
            help="DistilBERT is faster but slightly less accurate than BERT."
        )
    
    with col2:
        epochs = st.slider(
            "Number of epochs",
            min_value=1,
            max_value=10,
            value=3,
            help="More epochs generally means better accuracy but longer training time."
        )
    
    train_button = st.button("Train Model")
    
    if train_button:
        with st.spinner("Training model... This may take several minutes."):
            # Create progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Set up model
            detector = FakeJobDetector(model_type=model_type)
            
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
            
            progress_bar.progress(10)
            progress_text.text("Preprocessing data...")
            
            # Preprocess data
            df = detector.preprocess_data(df)
            
            progress_bar.progress(20)
            progress_text.text("Splitting dataset into train and test sets...")
            
            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fraudulent'])
            
            progress_bar.progress(30)
            progress_text.text(f"Training model for {epochs} epochs...")
            
            # Train model
            detector.train(train_df, epochs=epochs, batch_size=16)
            
            progress_bar.progress(80)
            progress_text.text("Evaluating model...")
            
            # Evaluate model
            evaluate_model(detector, test_df)
            
            progress_bar.progress(90)
            progress_text.text("Saving model...")
            
            # Save model
            detector.save_model('./fake_job_detector_model')
            
            progress_bar.progress(100)
            progress_text.text("Training complete!")
        
        st.success("Model training completed successfully!")
        st.info("Switch to the 'Job Analysis' tab to analyze job postings with your new model.")

if __name__ == "__main__":
    run_streamlit_app()
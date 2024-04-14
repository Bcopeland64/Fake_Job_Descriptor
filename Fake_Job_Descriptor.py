import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re

# Initialize tokenizer and model from the BERT pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# List of technical indicators relevant to technology job descriptions
tech_indicators = [
    'python', 'java', 'javascript', 'aws', 'azure', 'docker', 'kubernetes', 'react', 'angular', 'vue', 'node.js', 'sql',
    'nosql', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'machine learning', 'deep learning', 'api', 'rest',
    'graphql', 'devops', 'agile', 'scrum', 'blockchain', 'cybersecurity', 'iot', 'big data', 'hadoop', 'spark', 'git',
    'jenkins', 'linux', 'bash', 'C++', 'JavaScript', 'HTML', 'CSS', 'GitHub', 'CI/CD', 'Cloud Computing', 'Data Science',
    'Artificial Intelligence', 'Natural Language Processing', 'Computer Vision', 'Network Security', 'Cloud Storage',
    'Serverless Computing', 'Microservices', 'DevOps Automation', 'Continuous Integration', 'Continuous Deployment',
    'Containerization', 'Virtualization'
]

def text_cleaning(text):
    # Clean text by removing URLs, HTML tags, and non-alphanumeric characters, then convert to lower case
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(texts):
    # Preprocess text for BERT model input
    texts = [text_cleaning(text) for text in texts]
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

def analyze(text):
    # Analyze text to determine the likelihood of being genuine or fake
    inputs = preprocess_text([text])
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    fake_prob = probabilities[:, 1].item() * 100
    genuine_prob = probabilities[:, 0].item() * 100
    indicators_present = [word for word in tech_indicators if word in text_cleaning(text)]
    return genuine_prob, fake_prob, indicators_present

# Streamlit app setup for user interaction
st.title('Tech Job Ad Authenticity Checker')

# Manage the text area with session state for dynamic interaction
if 'text_area' not in st.session_state:
    st.session_state['text_area'] = ''

text = st.text_area("Enter a tech job description:", height=300, value=st.session_state.text_area, key='text_area')
analyze_button = st.button('Analyze')
clear_button = st.button('Clear', on_click=lambda: setattr(st.session_state, 'text_area', ''))  # Clear button functionality

if analyze_button and text:
    genuine_prob, fake_prob, indicators_present = analyze(text)
    # Determine which set of feedback to display based on higher probability
    if genuine_prob > fake_prob:
        st.success(f"This job description is likely genuine, based on the analysis with {genuine_prob:.2f}% probability:")
        genuine_feedback = [
            f"- Detected relevant technical terms: {', '.join(indicators_present)}.",
            "- A sufficient number of specific technical terms supports the legitimacy of this job ad."
        ] if len(indicators_present) >= 5 else [
            "- Although some technical terms are detected, always verify with the company directly for more details."
        ]
        for item in genuine_feedback:
            st.write(item)
    else:
        st.error(f"Warning: This job description is likely fake, based on the analysis with {fake_prob:.2f}% probability:")
        fake_feedback = [
            "- Financial transactions required upfront",
            "- Limited online presence of the company",
            "- Use of unofficial email domains",
            "- Unsolicited job offers are mentioned",
            "- Overpromising with unclear job responsibilities"
        ]  # Additional fake indicators can be appended here as needed
        for item in fake_feedback:
            st.write(item)
elif clear_button:
    st.info('The text has been cleared. You can now enter a new job description.')

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define tech-specific indicators
tech_indicators = [
    'python', 'java', 'javascript', 'aws', 'azure', 'docker', 'kubernetes',
    'react', 'angular', 'vue', 'node.js', 'sql', 'nosql', 'tensorflow', 'pytorch',
    'scikit-learn', 'pandas', 'machine learning', 'deep learning', 'api', 'rest',
    'graphql', 'devops', 'agile', 'scrum', 'blockchain', 'cybersecurity', 'iot',
    'big data', 'hadoop', 'spark', 'git', 'jenkins', 'linux', 'bash'
]

def text_cleaning(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(texts):
    texts = [text_cleaning(text) for text in texts]
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

def analyze(text):
    inputs = preprocess_text([text])
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    fake_prob = probabilities[:, 1].item() * 100
    genuine_prob = probabilities[:, 0].item() * 100
    indicators_present = [word for word in tech_indicators if word in text_cleaning(text)]
    return genuine_prob, fake_prob, indicators_present

# Streamlit app setup
st.title('Tech Job Ad Authenticity Checker')

# Using session state to manage the text area content
if 'text_area' not in st.session_state:
    st.session_state.text_area = ''

text = st.text_area("Enter a tech job description:", height=300, value=st.session_state.text_area, key='text_area')
analyze_button = st.button('Analyze')
clear_button = st.button('Clear', on_click=lambda: setattr(st.session_state, 'text_area', ''))

if analyze_button and text:
    genuine_prob, fake_prob, indicators_present = analyze(text)
    st.metric("Probability of being Genuine", f"{genuine_prob:.2f}%")
    st.metric("Probability of being Fake", f"{fake_prob:.2f}%")
    if fake_prob > 50:
        st.error("Warning: This job description is likely fake for several reasons:")
        feedback = []
        if 'high salary' in text.lower():
            feedback.append("- The mentioned salary is unusually high for the listed qualifications and responsibilities.")
        if 'immediate start' in text.lower():
            feedback.append("- Promises of immediate start without a proper interview process.")
        if len(indicators_present) < 3:
            feedback.append("- Lack of specific technical keywords that are common in genuine tech job descriptions.")
        if not feedback:
            feedback.append("- General indicators based on the analysis suggest this might be fraudulent.")
        for item in feedback:
            st.write(item)
    else:
        st.success("This job description is likely genuine, based on the analysis:")
        feedback = f"- Detected relevant technical terms: {', '.join(indicators_present)}."
        st.write(feedback)
        if len(indicators_present) >= 5:
            st.write("- A good number of specific technical terms are present which supports the legitimacy of this job ad.")
        else:
            st.write("- Although there are some technical terms present, always verify with the company directly for more details.")
elif clear_button:
    # Optional: Display a message confirming that the text has been cleared.
    st.info('The text has been cleared. You can now enter a new job description.')

# Fake_Job_Descriptor

# Tech Job Ad Authenticity Checker

This Streamlit-based web application is designed to help users determine the authenticity of tech job advertisements. By analyzing the content of a job description, the application provides insights into whether a job ad might be genuine or potentially fake.

## Features

- **Text Analysis:** Utilizes NLP to analyze job descriptions and identify key indicators of authenticity.
- **Tech-Specific Indicators:** Checks against a list of common tech-related terms to assess the relevance and specificity of the job ad.
- **Immediate Feedback:** Offers users immediate feedback on the potential authenticity of the job ad with detailed reasons.
- **Interactive UI:** Easy to use web interface built with Streamlit that allows users to input job descriptions and view analysis results.

## Technology Stack

- **Python**: Primary programming language.
- **Streamlit**: Framework for building the web app.
- **Transformers**: Library used for NLP tasks, specifically utilizing BERT models.
- **PyTorch**: Used in conjunction with Transformers for model handling.
- **Regex**: For text cleaning and preprocessing.

## Installation

Ensure you have Python installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

Follow these steps to set up and run the project locally:

## Clone the Repository

   ```
   git clone https://github.com/yourusername/tech-job-ad-authenticity-checker.git
   cd tech-job-ad-authenticity-checker
   
   ```

## Set Up a Virtual Environment (Optional but recommended)

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate

```
## Run the Application

```
streamlit run app.py

```

## Usage

Once the application is running:

- Enter a tech job description into the text area.
- Click the "Analyze" button to process the description.
- View the results displayed below the button, which indicate whether the job ad is likely genuine or fake, along with detailed explanations.


### Explanation
This README provides a structured overview covering:
- A brief description of the project and its features.
- Detailed setup instructions, including how to install Python, set up a virtual environment, install dependencies, and run the application.
- Guidelines on how users can utilize the app.
- Encouragement for community contributions including how to report issues and submit pull requests.
- License information to inform users of their rights regarding the use and distribution of the software.

Ensure to replace placeholders like the GitHub repository link `https://github.com/yourusername/tech-job-ad-authenticity-checker.git` with the actual URL to your project repository. This approach helps to make your project open and welcoming to potential users and contributors.


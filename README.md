# Emotion Detection from Text
Industry Project – TCS iON Internship

## Project Title
Automated Detection of Different Emotions from Textual Comments and Feedback

## Project Description
This project aims to build an automated system that detects and classifies human emotions from textual comments and feedback. The system analyzes unstructured text such as reviews, survey responses, and user comments to identify emotions like happiness, sadness, anger, fear, love, surprise, and neutral states.

The solution uses Natural Language Processing (NLP) techniques combined with Machine Learning to provide real-time emotion detection through a simple web-based interface.

## Objectives
- Analyze textual data and identify underlying emotions
- Preprocess and clean unstructured text data
- Extract features using NLP techniques
- Build and train a machine learning model for emotion classification
- Deploy the model using a user-friendly web application

## Technologies Used
- Programming Language: Python
- Libraries: Pandas, Scikit-learn, Joblib
- NLP Technique: TF-IDF Vectorization
- Machine Learning Model: Logistic Regression
- Frontend Framework: Streamlit
- Version Control: Git and GitHub

## Project Structure
emotion-detection-project
├── data
│   └── emotions.csv
├── model
│   └── train_model.py
├── app.py
├── requirements.txt
└── README.md

## Workflow
1. Data collection from emotion-labeled text sources
2. Text preprocessing and cleaning
3. Feature extraction using TF-IDF
4. Model training using Logistic Regression
5. Model testing and evaluation
6. Deployment using Streamlit web application

## Installation and Execution
Step 1: Install required libraries
pip install -r requirements.txt

Step 2: Train the model
python model/train_model.py

Step 3: Run the application
streamlit run app.py

## Sample Inputs and Outputs
Input: i am very happy today
Output: HAPPY

Input: this makes me angry
Output: ANGER

Input: i feel scared
Output: FEAR

Input: i love this product
Output: LOVE

Input: i am broken
Output: NEUTRAL / UNKNOWN

## Key Features
- Real-time emotion detection
- Simple and interactive user interface
- Handles unclear inputs using confidence threshold
- Hybrid approach combining ML and keyword-based logic

## Limitations
- Accuracy depends on dataset size and quality
- TF-IDF based models may struggle with sarcasm
- Limited language support

## Future Enhancements
- Use transformer-based models like BERT
- Add multilingual emotion detection
- Implement emotion intensity detection
- Improve accuracy with larger datasets
- Deploy the application on cloud platforms

## Internship Details
Organization: Tata Consultancy Services (TCS iON)
Project Type: Industry Internship Project
Domain: Artificial Intelligence / Natural Language Processing

## Conclusion
This project demonstrates the practical application of machine learning and NLP techniques in emotion detection. The system efficiently extracts emotional insights from textual feedback and can be applied in areas such as customer sentiment analysis, social media monitoring, and opinion mining.

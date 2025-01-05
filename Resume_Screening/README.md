# Resume Classifier

This is an AI-powered Resume Classifier application built with Streamlit. The app uses machine learning to classify resumes into different job categories based on their content. It supports PDF, DOCX, and TXT file formats.

## Features

- **Resume Upload**: Upload resumes in PDF, DOCX, or TXT formats.
- **Text Extraction**: Extracts the text from resumes using PyPDF2 for PDF files, python-docx for DOCX files, and custom encoding handling for TXT files.
- **Text Cleaning**: Cleans the extracted text by removing URLs, hashtags, mentions, and other irrelevant characters.
- **Prediction**: Classifies the resume into predefined job categories using a pre-trained machine learning model.
- **User-Friendly Interface**: Interactive web interface powered by Streamlit with file upload and category display.

## Technologies Used

- **Streamlit**: For creating the web interface.
- **Scikit-learn**: For machine learning and model predictions.
- **PyPDF2**: For extracting text from PDF files.
- **python-docx**: For extracting text from DOCX files.
- **Regex**: For cleaning and processing the text data.
- **Pickle**: For saving and loading the machine learning model, TF-IDF vectorizer, and label encoder.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/resume-classifier.git
   cd resume-classifier ```
   
2. **Install the dependencies:

Create a virtual environment and activate it:

```bash
Copy code
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate````

3. **Install the required libraries:

```bash
Copy code
pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run app.py```
This will open the app in your browser.

Usage
Upload a resume in PDF, DOCX, or TXT format using the file uploader on the main page.
The system will extract the text from the uploaded file.
The cleaned text will be displayed (optional).
The predicted job category will be shown.
Model Training
The machine learning model used to classify the resumes is a Support Vector Classifier (SVC) trained on a dataset of resumes. If you wish to retrain the model with your own data, you can use the following steps:

Preprocess the data (e.g., clean and vectorize the resume text).
Train the model using Scikit-learn's SVC.
Save the model, TF-IDF vectorizer, and label encoder using Pickle for later use.
Example Training Code:
python
Copy code
import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset (replace with your actual dataset)
# df = pd.read_csv('your_dataset.csv')

# Preprocess the text data (e.g., clean, vectorize)
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['resume_text'])
y = LabelEncoder().fit_transform(df['category'])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the model, vectorizer, and encoder
with open('clf.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(LabelEncoder(), f)
File Structure
graphql
Copy code
resume-classifier/
│
├── app.py               # Streamlit app for resume classification
├── clf.pkl              # Pre-trained SVC model file
├── tfidf.pkl            # Pre-trained TF-IDF vectorizer file
├── encoder.pkl          # Pre-trained label encoder file
├── requirements.txt     # List of Python dependencies
└── README.md            # Project documentation
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This project leverages machine learning techniques for text classification and natural language processing (NLP).
Special thanks to the open-source libraries that made this project possible: Streamlit, Scikit-learn, PyPDF2, python-docx, and Regex.
csharp
Copy code

### Instructions for GitHub Repository:

1. Replace `https://github.com/yourusername/resume-classifier.git` with your actual GitHub repository URL.
2. Add all necessary dependencies in the `requirements.txt` file. You can generate this file using:

   ```bash
   pip freeze > requirements.txt```

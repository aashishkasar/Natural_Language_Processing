# Resume Classifier

An AI-powered Resume Classifier application built with Streamlit. This tool uses machine learning to classify resumes into different job categories based on their content. It supports PDF, DOCX, and TXT file formats. 😊

## Features ✨

- **Resume Upload**: Upload resumes in PDF, DOCX, or TXT formats.
- **Text Extraction**: Extracts the text from resumes using:
  - 📄 PyPDF2 for PDF files.
  - 📝 python-docx for DOCX files.
  - 📃 Custom encoding handling for TXT files.
- **Text Cleaning**: Cleans extracted text by removing URLs, hashtags, mentions, and irrelevant characters.
- **Prediction**: Classifies resumes into predefined job categories using a pre-trained machine learning model. 🤖
- **User-Friendly Interface**: Interactive web interface powered by Streamlit for file upload and category display. 💻

## Technologies Used 🛠️

- **Streamlit**: For creating the web interface.
- **Scikit-learn**: For machine learning and model predictions.
- **PyPDF2**: For extracting text from PDF files.
- **python-docx**: For extracting text from DOCX files.
- **Regex**: For cleaning and processing text data.
- **Pickle**: For saving and loading the ML model, TF-IDF vectorizer, and label encoder.

## Installation 🚀

### Clone the repository

```bash
git clone https://github.com/<yourusername>/resume-classifier.git
cd resume-classifier
```

### Create a virtual environment and activate it

```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

### Install the dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

This will open the app in your browser. 🌐

## Usage 📂

1. Upload a resume in PDF, DOCX, or TXT format using the file uploader on the main page.
2. The system extracts the text from the uploaded file.
3. The cleaned text (optional) and the predicted job category are displayed. 🏷️

## Model Training 🤓

The machine learning model used to classify resumes is a Support Vector Classifier (SVC) trained on a dataset of resumes.

### To retrain the model:

1. **Preprocess the data** (clean and vectorize the resume text).
2. **Train the model** using Scikit-learn's SVC.
3. **Save the model** and associated tools (TF-IDF vectorizer, label encoder) using Pickle.

### Example Training Code

```python
import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset (replace with your dataset)
# df = pd.read_csv('your_dataset.csv')

# Preprocess the text data
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
```

## File Structure 🗂️

```plaintext
resume-classifier/
├── app.py               # Streamlit app for resume classification
├── clf.pkl              # Pre-trained SVC model file
├── tfidf.pkl            # Pre-trained TF-IDF vectorizer file
├── encoder.pkl          # Pre-trained label encoder file
├── requirements.txt     # List of Python dependencies
└── README.md            # Project documentation
```

## Acknowledgments 🙏

This project leverages machine learning techniques for text classification and natural language processing (NLP). Special thanks to the open-source libraries that made this project possible:

- Streamlit
- Scikit-learn
- PyPDF2
- python-docx
- Regex

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


import streamlit as st
import pickle
import docx
import PyPDF2
import re
from PIL import Image

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Update with your model path
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Update with your vectorizer path
le = pickle.load(open('encoder.pkl', 'rb'))  # Update with your encoder path


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


def main():

    st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="wide")

    # Sidebar design
    # st.sidebar.image("sidebar_logo.png", use_column_width=True)  # Add your sidebar logo
    st.sidebar.title("Navigation")
    st.sidebar.write("👋 Welcome to the Resume Classifier!")
    st.sidebar.info("Use this tool to predict the category of resumes.")
    st.sidebar.markdown("---")
    st.sidebar.header("Instructions")
    st.sidebar.write("1. Upload a resume file (PDF, DOCX, or TXT).")
    st.sidebar.write("2. View the extracted resume text.")
    st.sidebar.write("3. Get the predicted job category.")

    # Main page
    st.title("📄 Resume Classifier")
    st.markdown("Upload your resume and get an AI-powered prediction of the job category.")

    # File upload
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("Successfully extracted the text from the uploaded resume.")
            
            # Display extracted text
            with st.expander("View Extracted Text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Display prediction
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"The predicted category is: **{category}**")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Footer with copyright and developer credits
    st.markdown("---")
    col1, col2 = st.columns([1, 3])

   

    with col2:
        st.markdown(
            """
            <p style='text-align: center;'>
            &copy; 2025 Resume Classifier. All rights reserved.<br>
            Developed with ❤️ by <a href="https://github.com/aashishkasar" target="_blank">Aashish</a>.
            </p>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()

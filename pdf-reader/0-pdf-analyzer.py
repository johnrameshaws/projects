import streamlit as st
import os
import re
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List
from pypdf import PdfReader

#from google import genai
#from google.genai import types
#import httpx  

import shutil
from tempfile import NamedTemporaryFile

#os.environ['GOOGLE_API_KEY'] = "AIzaSyCmQy3CSLnSanxsLvss0l3qPE-rYK7wJUo" #st.secrets['GEMINI_KEY']

#client = genai.Client(os.environ['GOOGLE_API_KEY'])
# Set and validate the API key for Gemini API
os.environ['GEMINI_API_KEY'] = 'AIzaSyCmQy3CSLnSanxsLvss0l3qPE-rYK7wJUo'
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Gemini API Key not provided or incorrect. Please provide a valid GEMINI_API_KEY.")
try:
    genai.configure(api_key=gemini_api_key)
    print("API configured successfully with the provided key.")
except Exception as e:
    print("Failed to configure API:", str(e))

def save_uploaded_file(uploaded_file):
    with NamedTemporaryFile(dir='.', suffix='.pdf', delete=False) as f:
        f.write(uploaded_file.getbuffer())
        return f.name

def analyze_image(image_path):
    with open(image_path, "rb") as f:
        pdf_data = f.read()

    # Create a prompt for the model
    prompt = "Summarize this document in detail in json format:"

    with st.spinner('Analyzing image...'):
        # Send the PDF data to the model
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Use the Gemini 2.0 Flash model
            contents=[
                prompt,
                types.Part.from_bytes(data=pdf_data, mime_type="application/pdf")
            ],
        )
        st.markdown(response.text)

# Load the PDF file and extract text from each page
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Split the text into chunks based on double newlines
def split_text(text):
    return [i for i in re.split('\n\n', text) if i.strip()]

# Define a custom embedding function using Gemini API
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

# Create a Chroma database with the given documents
def create_chroma_db(documents: List[str], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    for i, d in enumerate(documents):
        db.add(documents=[d], ids=[str(i)])
    return db, name

# Load an existing Chroma collection
def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

# Retrieve the most relevant passages based on the query
def get_relevant_passage(query: str, db, n_results: int):
    results = db.query(query_texts=[query], n_results=n_results)
    return [doc[0] for doc in results['documents']]


# Construct a prompt for the generation model based on the query and retrieved data
def make_rag_prompt(query: str, relevant_passage: str):
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
    strike a friendly and conversational tone.
    QUESTION: '{query}'
    PASSAGE: '{escaped_passage}'

    ANSWER:
    """
    return prompt

# Generate an answer using the Gemini Pro API
def generate_answer(prompt: str):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    result = model.generate_content(prompt)
    return result.text

# Interactive function to process user input and generate an answer
def process_query_and_generate_answer(query):
    """
    query = input("Please enter your query: ")
    if not query:
        print("No query provided.")
        return
    """

    db = load_chroma_collection(db_path, db_name)
    relevant_text = get_relevant_passage(query, db, n_results=1)
    if not relevant_text:
        print("No relevant information found for the given query.")
        return
    final_prompt = make_rag_prompt(query, "".join(relevant_text))
    answer = generate_answer(final_prompt)
    print("Generated Answer:", answer)

    db_folder = "chroma_db"
    db_name = "rag_experiment"

def load_pdf_and_load_db(pdf_path, db_name):
    with st.spinner('Analyzing pdf file...'):
        if os.path.exists(db_folder):
            shutil.rmtree(db_folder)
        
        os.makedirs(db_folder)

        # Specify the path and collection name for Chroma database
        db_path = os.path.join(os.getcwd(), db_folder)

        pdf_text = load_pdf(pdf_path)
        chunked_text = split_text(pdf_text)

        print(chunked_text, db_path, db_name)
        
        db, db_name = create_chroma_db(chunked_text, db_path, db_name)

        db = load_chroma_collection(db_path, db_name)

        return db

def main():
    st.title("🔍 PDF Analyzer")
    with st.sidebar:
        st.title("Settings")

    tab_examples, tab_upload, tab_camera = st.tabs([
        "📚 Example Products", 
        "📤 Upload Image", 
        "📸 Take Photo"
    ])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload pdf file", 
            type=["pdf"],
            help="Upload a pdf file"
        )

        if uploaded_file:
            if st.button("🔍 Analyze Uploaded pdf", key="analyze_upload"):
                temp_path = save_uploaded_file(uploaded_file)
                load_pdf_and_load_db(temp_path, db_name)
                #analyze_image(temp_path)
                question=st.text_area("text to analyze", "What is the AI Maturity Scale?")
                if st.button("Generate Answer"):
                    process_query_and_generate_answer(question)

                os.unlink(temp_path) 

if __name__ == "__main__":
    st.set_page_config(
        page_title="Product Ingredient Agent",
        layout="wide",
        #initial_sidebar_state="collapsed"
    )

    st.markdown(
        """
        <style>
        .st-emotion-cache-1jicfl2 {
            width: 100%;
            padding: 6rem 1rem 10rem;
            min-width: auto;
            max-width: initial;
        }
        """,
        unsafe_allow_html=True,
    )

    db_folder = "chroma_db"
    db_name = "rag_experiment"
    db_path=""

    main()

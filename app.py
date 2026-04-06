import streamlit as st
st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖", layout="centered")
# UI
st.title("AI Knowledge Assistant (RAG Chatbot)")
st.caption("💬 Ask anything or upload a PDF to chat with your document")
st.divider()
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
# Reset chat if new file uploaded
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None
if uploaded_file and uploaded_file != st.session_state.last_uploaded:
    st.session_state.last_uploaded = uploaded_file
from dotenv import load_dotenv
import os
import ssl
import certifi
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
load_dotenv()
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from pypdf import PdfReader
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text
# Load API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
# Gemini LLM
model = genai.GenerativeModel("models/gemini-flash-latest")
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vector_db(file):
    # Load data
    if file:
        text = extract_text_from_pdf(file)
    else:
        with open("data.txt", "r") as file:
            text = file.read()
    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    docs = text_splitter.split_text(text)
    # Local embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    # Vector DB
    db = Chroma.from_texts(docs, embeddings)
    return db
db = load_vector_db(uploaded_file)
# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Query input
query = st.chat_input("Ask something...")
if query:
    with st.spinner("Thinking..."):
        results = db.similarity_search(query)
    context = " ".join([doc.page_content for doc in results])
    prompt = f"""
    Answer based on the context below:
    Context:
    {context}
    Question:
    {query}
    """
    response = model.generate_content(prompt)
    answer = response.text
    # Save chat
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("Bot", answer))
# Display chat history
# Display chat properly
for role, message in st.session_state.chat_history:
    with st.chat_message("user" if role == "You" else "assistant"):
        st.write(message)
# for role, message in st.session_state.chat_history:
#     if role == "You":
#         st.markdown(f"""
#         <div style='background-color:#DCF8C6;padding:10px;border-radius:10px;margin:5px 0;'>
#         <b>You:</b> {message}
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#         <div style='background-color:#F1F0F0;padding:10px;border-radius:10px;margin:5px 0;'>
#         <b>AI:</b> {message}
#         </div>
#         """, unsafe_allow_html=True)
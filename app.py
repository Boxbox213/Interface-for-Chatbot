import streamlit as st
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="USeP Campus FAQ Chatbot", page_icon="🎓")

st.title("🎓 USeP Campus FAQ Chatbot")
st.markdown("Ask me anything about USeP based on the Pre-Enrollment Procedure and Student Handbook.")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    st.markdown("Get your Groq API key from [Groq Console](https://console.groq.com/keys)")

@st.cache_resource
def load_and_process_documents():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        return None
    
    raw_documents = []
    for filename in pdf_files:
        loader = PyPDFLoader(filename)
        raw_documents.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    documents = text_splitter.split_documents(raw_documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

with st.spinner("Loading and processing documents (this may take a moment on first run)..."):
    retriever = load_and_process_documents()

if not retriever:
    st.warning("No PDF documents found in the current directory. Please place your PDFs here.")
    st.stop()

if not groq_api_key:
    st.info("Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

# Initialize ChatGroq and chain
try:
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_api_key
    )
except Exception as e:
    st.error(f"Error initializing Groq: {e}")
    st.stop()

system_prompt = (
    "You are a specialized AI assistant for the USeP (University of Southeastern Philippines) "
    "Campus FAQ domain.\n"
    "Answer questions strictly using ONLY the provided context below.\n"
    "If the answer cannot be found in the context, reply: 'I cannot answer based on the provided "
    "domain data.'\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about USeP..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = retrieval_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")

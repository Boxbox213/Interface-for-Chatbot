import streamlit as st
import os
import glob
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="USeP Campus FAQ Chatbot", page_icon="🎓", layout="centered")

st.title("🎓 USeP Campus FAQ Chatbot")
st.markdown("Ask me anything about USeP based on the Pre-Enrollment Procedure and Student Handbook.")

# Sidebar Settings
with st.sidebar:
    with st.expander("⚙️ Settings", expanded=True):
        groq_api_key = st.text_input("Groq API Key", type="password")
        groq_model = st.text_input("Groq Model ID", value="openai/gpt-oss-20b")
        st.markdown("Get your key: [Groq Console](https://console.groq.com/keys)")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

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
        model_name=groq_model,
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

# Typing animation generator
def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message and chips if history is empty
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome to the USeP Campus FAQ Chatbot! Ask a question to get started.")
    col1, col2 = st.columns(2)
    example_query = None
    with col1:
        if st.button("📝 What are the pre-enrollment requirements?", use_container_width=True):
            example_query = "What are the pre-enrollment requirements?"
    with col2:
        if st.button("⚖️ What are the academic policies?", use_container_width=True):
            example_query = "What are the academic policies?"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for doc in message["sources"]:
                    source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**🏷️ {source_file}** (Page {page})")
                    st.caption(doc.page_content)

user_input = st.chat_input("Ask a question about USeP...")

# If an example chip was clicked, process it as user input
if len(st.session_state.messages) == 0 and 'example_query' in locals() and example_query:
    user_input = example_query

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Retrieving knowledge..."):
            try:
                response = retrieval_chain.invoke({"input": user_input})
                answer = response["answer"]
                docs = response.get("context", [])
                
                # Typing effect
                st.write_stream(stream_text(answer))
                
                # Sources display
                if docs:
                    with st.expander("📚 Sources"):
                        for doc in docs:
                            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page = doc.metadata.get("page", "?")
                            st.markdown(f"**🏷️ {source_file}** (Page {page})")
                            st.caption(doc.page_content)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": docs
                })
            except Exception as e:
                st.error(f"Error generating response: {e}")

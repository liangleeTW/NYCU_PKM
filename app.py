__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
from datetime import datetime
import os

from rag import RAGSystem
from utils import generate_user_id, generate_session_id, save_uploaded_file


# Initialize RAG system with selected model
@st.cache_resource
def get_rag_system(_model_name):
    return RAGSystem(model_name=_model_name)

# Get RAG system with selected model
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o"

rag_system = get_rag_system(st.session_state.selected_model)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = generate_user_id()

if "session_id" not in st.session_state:
    st.session_state.session_id = generate_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App title
st.title("RAG Chat System")

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose documents", accept_multiple_files=True, type=["pdf", "txt", "csv"])
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                # Check if this file was already uploaded
                if uploaded_file.name not in [doc["filename"] for doc in st.session_state.uploaded_documents]:
                    # Save the uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Save document info to session state
                    st.session_state.uploaded_documents.append({
                        "filename": uploaded_file.name,
                        "file_path": file_path,
                        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Process the document with the RAG system
                    documents = rag_system.load_document(file_path)
                    rag_system.add_documents(documents)
                    
                    st.success(f"✅ Added {uploaded_file.name} to knowledge base!")
    
    # Display user documents
    st.header("Your Documents")
    if st.session_state.uploaded_documents:
        for doc in st.session_state.uploaded_documents:
            st.text(f"📄 {doc['filename']} ({doc['uploaded_at']})")
    else:
        st.info("No documents uploaded yet.")
    
    # Option to start a new session
    if st.button("Start New Chat"):
        st.session_state.session_id = generate_session_id()
        st.session_state.messages = []
        st.rerun()

# Model selection
st.header("Model Selection")
model_options = [
    "gpt-4o",
    "gpt-4o-mini", 
    "gpt-4-turbo",
    "gpt-3.5-turbo"
]
selected_model = st.selectbox(
    "Choose AI Model:",
    model_options,
    index=0,  # Default to gpt-4o
    help="Select the OpenAI model for generating responses"
)

# Store selected model in session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = selected_model
else:
    st.session_state.selected_model = selected_model
    
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate a response
    with st.chat_message("assistant"):
        # with st.spinner("Thinking..."):
        # Get chat history
        history = [
            {"query": msg["content"], "response": st.session_state.messages[i+1]["content"]}
            for i, msg in enumerate(st.session_state.messages[:-1])
            if msg["role"] == "user" and i+1 < len(st.session_state.messages)
        ]
        
        # Create placeholder for streaming response
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in rag_system.generate_response(prompt, history):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        # Final response without cursor
        response_placeholder.markdown(full_response)
            
        # Save chat to session state
        st.session_state.chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": prompt,
            "response": full_response
        })
    
    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Display chat history tab
st.header("Chat History")

if st.session_state.chat_history:
    # Create a dataframe from chat history
    history_df = pd.DataFrame([
        {
            "Timestamp": chat["timestamp"],
            "Query": chat["query"],
            "Response": chat["response"]
        }
        for chat in st.session_state.chat_history
    ])
    
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No chat history for this session yet.")


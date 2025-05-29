import os
import uuid
import tempfile
from typing import List, Dict, Any
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
CHROMA_PERSIST_DIRECTORY = os.path.join(tempfile.gettempdir(), "chroma_db")
COLLECTION_NAME = "documents"

class RAGSystem:
    def __init__(self, model_name: str = "gpt-4o"):
        # Store the model name
        self.model_name = model_name
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize ChromaDB
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        self.vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
            
        # Initialize LLM with selected model
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.2)
        self.model_name = model_name
        
        # Create text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document and split it into chunks"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Choose the appropriate loader based on file extension
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
        else:
            # Default to text loader for other file types
            loader = TextLoader(file_path)
        
        # Load the document
        documents = loader.load()
        
        # Split the document into chunks
        return self.text_splitter.split_documents(documents)
    
    def add_documents(self, documents: List[Document]) -> str:
        """Add documents to the vector store"""
        # Generate a unique collection ID
        collection_id = str(uuid.uuid4())
        
        # Add documents to the vector store
        self.vector_store.add_documents(documents)
        
        # Persist the vector store
        self.vector_store.persist()
        
        return collection_id
    
    def retrieve_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for a query"""
        return self.vector_store.similarity_search(query, k=k)
    
    def generate_response(self, query: str, history: List[Dict[str, str]] = None) -> str:
        """Generate a response for a query using RAG"""
        # Create a prompt template that includes chat history
        template = """
        You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Format chat history for reference (not directly passed to the chain)
        chat_history_str = ""
        if history:
            for entry in history:
                chat_history_str += f"User: {entry['query']}\nAssistant: {entry['response']}\n\n"
        
        # Create the prompt template with the correct input variables
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Retrieve relevant documents
        docs = self.retrieve_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
    
        # Create the full prompt
        full_prompt = prompt.format(context=context, question=query)
        
        # Stream the response
        for chunk in self.llm.stream(full_prompt):
            if chunk.content:
                yield chunk.content
                
        # Generate a response
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Use the chain with the correct input key 'query'
        response = chain({"query": query})
        
        return response["result"]

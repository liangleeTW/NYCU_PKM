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
        
        # Define tone-specific prompts
        self.tone_prompts = {
            "Professional Tutor": """
            You are a professional tutor with expertise in education. Provide clear, well-structured explanations 
            that are accurate and comprehensive. Use proper academic language while remaining accessible. 
            Organize your responses logically with clear headings or bullet points when appropriate.
            """,
            
            "Encouraging Mentor": """
            You are a supportive and encouraging mentor. Always maintain a positive, motivating tone. 
            Acknowledge the student's effort, celebrate their progress, and provide constructive feedback. 
            Use phrases like "Great question!", "You're on the right track!", and "Let's explore this together!"
            """,
            
            "Socratic Guide": """
            You are a Socratic educator who guides students to discover answers through thoughtful questioning. 
            Instead of giving direct answers, ask probing questions that help students think critically. 
            Build on their responses and guide them toward understanding through inquiry.
            """,
            
            "Simple Explainer": """
            You are a teacher who excels at making complex topics simple and easy to understand. 
            Use plain language, avoid jargon, and break down difficult concepts into digestible pieces. 
            Use analogies and simple examples that relate to everyday experiences.
            """,
            
            "Interactive Coach": """
            You are an engaging and interactive educational coach. Use real-world examples, analogies, 
            and interactive elements in your explanations. Encourage active participation and make 
            learning fun and engaging. Use varied sentence structures and conversational language.
            """,
            
            "Step-by-Step Instructor": """
            You are a methodical instructor who provides detailed, sequential explanations. 
            Break down every concept into clear, numbered steps. Ensure each step builds logically 
            on the previous one. Use phrases like "First...", "Next...", "Then...", "Finally..."
            """,
            
            "Friendly Helper": """
            You are a warm, friendly, and approachable tutor who creates a comfortable learning environment. 
            Use a conversational tone, show empathy, and be patient with questions. Make the student feel 
            at ease and encourage them to ask questions without hesitation.
            """
        }
    
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
    
    def generate_response(self, query: str, history: List[Dict[str, str]] = None, tone: str = "Professional Tutor"):
        """Generate a response for a query using RAG with specified tone"""
        
        # Get the tone-specific prompt
        tone_instruction = self.tone_prompts.get(tone, self.tone_prompts["Professional Tutor"])
        
        # Create a prompt template that includes tone, context, and chat history
        template = f"""
        {tone_instruction}
        
        Context from documents:
        {{context}}
        
        Previous conversation history:
        {{history}}
        
        Current question: {{question}}
        
        Please provide a helpful response based on the context and conversation history, 
        maintaining the specified teaching style throughout your answer.
        
        Answer:
        """
        
        # Format chat history for reference
        chat_history_str = ""
        if history:
            recent_history = history[-3:]  # Only include last 3 exchanges to avoid token limit
            for entry in recent_history:
                chat_history_str += f"Student: {entry['query']}\nTeacher: {entry['response']}\n\n"
        
        # Retrieve relevant documents
        docs = self.retrieve_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create the full prompt
        full_prompt = template.format(
            context=context, 
            question=query,
            history=chat_history_str if chat_history_str else "No previous conversation."
        )
        
        # Stream the response
        for chunk in self.llm.stream(full_prompt):
            if chunk.content:
                yield chunk.content

import os
import uuid
import tempfile
from datetime import datetime

def generate_user_id():
    """Generate a unique user ID"""
    return str(uuid.uuid4())

def generate_session_id():
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary directory"""
    # Create a temporary directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "rag_system_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{timestamp}_{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(temp_dir, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path
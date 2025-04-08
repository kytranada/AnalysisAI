import os
import uuid
import tempfile
import shutil
from typing import Dict, Optional, List, Tuple
import pandas as pd

class FileManager:
    """
    Manages uploaded files, their metadata, and associated DataFrames.
    Provides a more robust alternative to using global dictionaries.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the FileManager.
        
        Args:
            storage_dir: Directory to store uploaded files. If None, a temporary directory will be used.
        """
        if storage_dir:
            self.storage_dir = storage_dir
            os.makedirs(storage_dir, exist_ok=True)
        else:
            # Create a temporary directory for file storage
            self.storage_dir = tempfile.mkdtemp(prefix="duckdb_uploads_")
        
        # Dictionary to store file metadata by ID
        self.files: Dict[str, Dict] = {}
        
        # Dictionary to cache DataFrames by ID (lazy loading)
        self._dataframes: Dict[str, pd.DataFrame] = {}
    
    def save_uploaded_file(self, file_content: bytes, original_filename: str) -> Tuple[str, str]:
        """
        Save an uploaded file and return its ID and path.
        
        Args:
            file_content: The content of the uploaded file
            original_filename: The original filename
            
        Returns:
            Tuple of (file_id, file_path)
        """
        # Generate a unique ID for the file
        file_id = str(uuid.uuid4())
        
        # Get the file extension
        _, file_extension = os.path.splitext(original_filename)
        
        # Create the file path
        file_path = os.path.join(self.storage_dir, f"{file_id}{file_extension}")
        
        # Write the file content
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Store file metadata
        self.files[file_id] = {
            'id': file_id,
            'original_filename': original_filename,
            'path': file_path,
            'extension': file_extension,
        }
        
        return file_id, file_path
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get the path of a file by ID."""
        if file_id in self.files:
            return self.files[file_id]['path']
        return None
    
    def get_file_metadata(self, file_id: str) -> Optional[Dict]:
        """Get the metadata of a file by ID."""
        return self.files.get(file_id)
    
    def get_dataframe(self, file_id: str, load_func) -> Optional[pd.DataFrame]:
        """
        Get the DataFrame for a file by ID.
        If the DataFrame is not cached, it will be loaded using the provided load function.
        
        Args:
            file_id: The ID of the file
            load_func: A function that takes a file path and returns a DataFrame
            
        Returns:
            The DataFrame or None if the file doesn't exist
        """
        # Check if the file exists
        file_path = self.get_file_path(file_id)
        if not file_path:
            return None
        
        # Check if the DataFrame is cached
        if file_id not in self._dataframes:
            try:
                # Load the DataFrame
                self._dataframes[file_id] = load_func(file_path)
            except Exception as e:
                print(f"Error loading DataFrame for file {file_id}: {e}")
                return None
        
        return self._dataframes[file_id]
    
    def clear_dataframe_cache(self, file_id: Optional[str] = None):
        """
        Clear the DataFrame cache for a specific file or all files.
        
        Args:
            file_id: The ID of the file to clear, or None to clear all
        """
        if file_id:
            if file_id in self._dataframes:
                del self._dataframes[file_id]
        else:
            self._dataframes.clear()
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file by ID.
        
        Args:
            file_id: The ID of the file to delete
            
        Returns:
            True if the file was deleted, False otherwise
        """
        if file_id not in self.files:
            return False
        
        # Get the file path
        file_path = self.files[file_id]['path']
        
        # Delete the file
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_id}: {e}")
            return False
        
        # Remove from dictionaries
        del self.files[file_id]
        if file_id in self._dataframes:
            del self._dataframes[file_id]
        
        return True
    
    def list_files(self) -> List[Dict]:
        """List all files with their metadata."""
        return list(self.files.values())
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        # Clear the DataFrame cache
        self._dataframes.clear()
        
        # Delete all files
        for file_id in list(self.files.keys()):
            self.delete_file(file_id)
        
        # Remove the storage directory if it's a temporary one
        if os.path.exists(self.storage_dir) and self.storage_dir.startswith(tempfile.gettempdir()):
            try:
                shutil.rmtree(self.storage_dir)
            except Exception as e:
                print(f"Error removing storage directory: {e}")

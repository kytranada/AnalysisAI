import os
import math
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import duckdb
from dotenv import load_dotenv

# Import from core modules
from core.metadata import ColumnMetadataManager
from core.data_utils import load_data, get_db_schema, sanitize_data_for_json
from core.query_engine import execute_query
from core.file_manager import FileManager

# --- Global instances ---
# Create a single instance of FileManager and ColumnMetadataManager to be shared across requests
file_manager = FileManager()
column_manager = ColumnMetadataManager()

# --- Models ---
class QueryRequest(BaseModel):
    query: str
    file_id: str

class QueryResponse(BaseModel):
    answer: str
    sql_query: Optional[str] = None
    execution_result: Optional[str] = None

class StatsResponse(BaseModel):
    row_count: int
    column_count: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    summary_stats: Dict[str, Any]

# --- Application ---
# Define the lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    # Startup: Nothing special to do as we've already created our global instances
    print("API server starting up...")
    yield
    # Shutdown: Clean up resources
    try:
        file_manager.cleanup()
        print("Cleaned up file manager resources.")
    except Exception as e:
        print(f"Error cleaning up file manager: {e}")

# Create the FastAPI app with the lifespan
app = FastAPI(title="DuckDB Query API", lifespan=lifespan)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency Injection ---
def get_llm():
    """Get the LLM instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Load environment variables if not already loaded
    load_dotenv()
    
    # Get the API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not found in environment variables")
    
    # Create the LLM instance
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-001", google_api_key=google_api_key)

def get_column_manager():
    """Get the ColumnMetadataManager instance."""
    # Return the global instance instead of creating a new one
    global column_manager
    return column_manager

def get_file_manager():
    """Get the FileManager instance."""
    # Return the global instance instead of creating a new one
    global file_manager
    return file_manager

def get_db_connection():
    """Get the DuckDB connection."""
    return duckdb.connect(database=':memory:', read_only=False)


# --- Helper Functions ---
def get_column_metadata_string(column_manager: ColumnMetadataManager, dataset_id: str) -> str:
    """Get the column metadata string for a dataset."""
    metadata = column_manager.dataset_metadata.get(dataset_id)
    if not metadata:
        metadata = column_manager.load_metadata(dataset_id)
        if not metadata:
            return ""
    
    # Format column metadata as a string
    metadata_str = "Column Metadata:\n"
    for col_name, col_info in metadata["columns"].items():
        metadata_str += f"- {col_name}: {col_info['description']}\n"
        if col_info['synonyms']:
            metadata_str += f"  Synonyms: {', '.join(col_info['synonyms'])}\n"
    
    return metadata_str

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "DuckDB Query API is running"}

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_manager: FileManager = Depends(get_file_manager),
    column_manager: ColumnMetadataManager = Depends(get_column_manager),
    llm = Depends(get_llm),
    connection = Depends(get_db_connection)
):
    """Upload a CSV or Excel file for analysis."""
    try:
        # Check file extension
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in ['.csv', '.xlsx', '.xls']:
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Read the file content
        content = await file.read()
        
        # Save the file
        file_id, file_path = file_manager.save_uploaded_file(content, file.filename)
        
        # Load the data
        df = load_data(file_path)
        
        # Extract column metadata
        column_manager.extract_column_metadata(df, file_path, llm)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query(
    request: QueryRequest,
    file_manager: FileManager = Depends(get_file_manager),
    column_manager: ColumnMetadataManager = Depends(get_column_manager),
    llm = Depends(get_llm),
    connection = Depends(get_db_connection)
):
    """Execute a natural language query against the uploaded file."""
    try:
        file_id = request.file_id
        query_text = request.query
        
        # Get the file path
        file_path = file_manager.get_file_path(file_id)
        if not file_path:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
        
        # Get the DataFrame
        df = file_manager.get_dataframe(file_id, load_data)
        if df is None:
            raise HTTPException(status_code=500, detail=f"Error loading file with ID {file_id}")
        
        # Make the DataFrame available to DuckDB
        connection.register("user_df", df)
        
        # Get the schema
        schema_string = get_db_schema(connection, "user_df")
        
        # Get column metadata
        dataset_id = os.path.basename(file_path)
        column_metadata_string = get_column_metadata_string(column_manager, dataset_id)
        
        # Preprocess the query
        processed_query = column_manager.preprocess_query(query_text, dataset_id)
        
        # Execute the query using our optimized function
        result = execute_query(
            processed_query, 
            llm, 
            connection, 
            schema_string, 
            column_metadata_string
        )
        
        # Sanitize the execution result to ensure it's JSON-compliant
        sanitized_execution_result = sanitize_data_for_json(result.execution_result)
        
        return {
            "answer": result.answer,
            "sql_query": result.sql_query,
            "execution_result": sanitized_execution_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/{file_id}")
async def get_stats(
    file_id: str,
    file_manager: FileManager = Depends(get_file_manager)
):
    """Get statistics for the uploaded file."""
    try:
        # Get the DataFrame
        df = file_manager.get_dataframe(file_id, load_data)
        if df is None:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
        
        # Calculate basic statistics
        row_count = len(df)
        column_count = len(df.columns)
        
        # Identify numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Calculate summary statistics
        summary_stats = {}
        
        # For numeric columns
        if numeric_columns:
            numeric_stats = df[numeric_columns].describe().to_dict()
            # Convert numpy types to Python native types for JSON serialization
            for col, stats in numeric_stats.items():
                summary_stats[col] = {k: float(v) if pd.notna(v) else None for k, v in stats.items()}
        
        # For categorical columns
        for col in categorical_columns:
            value_counts = df[col].value_counts().nlargest(5).to_dict()
            summary_stats[col] = {
                "top_values": {str(k): int(v) for k, v in value_counts.items()}
            }
        
        # Sanitize the data to ensure it's JSON-compliant
        sanitized_summary_stats = sanitize_data_for_json(summary_stats)
        
        return {
            "row_count": row_count,
            "column_count": column_count,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "summary_stats": sanitized_summary_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metadata/{file_id}")
async def get_metadata(
    file_id: str,
    file_manager: FileManager = Depends(get_file_manager),
    column_manager: ColumnMetadataManager = Depends(get_column_manager),
    llm = Depends(get_llm)
):
    """Get column metadata for the uploaded file."""
    try:
        # Get the file path
        file_path = file_manager.get_file_path(file_id)
        if not file_path:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
        
        # Get the dataset ID
        dataset_id = os.path.basename(file_path)
        
        # Load metadata if not already loaded
        if dataset_id not in column_manager.dataset_metadata:
            metadata = column_manager.load_metadata(dataset_id)
            if not metadata:
                # If metadata doesn't exist, extract it
                df = file_manager.get_dataframe(file_id, load_data)
                if df is None:
                    raise HTTPException(status_code=500, detail=f"Error loading file with ID {file_id}")
                metadata = column_manager.extract_column_metadata(df, file_path, llm)
        else:
            metadata = column_manager.dataset_metadata[dataset_id]
        
        # Return the metadata
        return {"metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/{file_id}")
async def get_data(
    file_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    file_manager: FileManager = Depends(get_file_manager)
):
    """
    Get paginated data for the uploaded file.
    
    Args:
        file_id: The ID of the file
        page: The page number (1-indexed)
        page_size: The number of rows per page
    """
    try:
        # Get the DataFrame
        df = file_manager.get_dataframe(file_id, load_data)
        if df is None:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
        
        # Calculate pagination
        total_rows = len(df)
        total_pages = math.ceil(total_rows / page_size)
        
        # Validate page number
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        # Get the slice of data for this page
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        # Get the data for this page
        page_data = df.iloc[start_idx:end_idx].to_dict(orient='records')
        
        # Sanitize the data to ensure it's JSON-compliant
        sanitized_data = sanitize_data_for_json(page_data)
        
        return {
            "data": sanitized_data,
            "columns": df.columns.tolist(),
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_rows": total_rows,
                "total_pages": total_pages
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/preview/{file_id}")
async def get_preview(
    file_id: str,
    limit: int = Query(10, ge=1, le=100),
    file_manager: FileManager = Depends(get_file_manager)
):
    """Get a preview of the uploaded file."""
    try:
        # Get the DataFrame
        df = file_manager.get_dataframe(file_id, load_data)
        if df is None:
            raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
        
        # Get the first N rows
        preview = df.head(limit).to_dict(orient='records')
        
        # Sanitize the preview data to ensure it's JSON-compliant
        sanitized_preview = sanitize_data_for_json(preview)
        
        return {"preview": sanitized_preview, "columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Run the application ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import pandas as pd
import duckdb
from io import StringIO
from typing import Optional

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV or Excel file into a pandas DataFrame."""
    print(f"Attempting to load data from: {file_path}")
    _, file_extension = os.path.splitext(file_path)

    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Please use CSV or Excel.")

        print("Data loaded successfully.")
        # Optional: Clean column names (replace spaces, special chars) if needed
        df.columns = ["_".join(col.lower().split()) for col in df.columns]
        print("Column names cleaned (lowercase, spaces replaced with underscores).")

        print("\n--- DataFrame Head ---")
        print(df.head())
        print("\n--- DataFrame Info ---")
        df.info()
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def get_db_schema(connection, df_variable_name: str) -> str:
    """Gets the table schema string using DuckDB introspection."""
    print(f"\nGetting schema for DataFrame variable: {df_variable_name}")
    try:
        # Use DESCRIBE on the DataFrame variable name
        schema_df = connection.execute(f"DESCRIBE {df_variable_name};").df()
        schema_string = schema_df.to_string()
        print("Schema retrieved using DESCRIBE.")
    except Exception as e1:
        print(f"DESCRIBE failed ({e1}), trying fallback...")
        try:
            # Fallback: Create a basic schema string from pandas info
            # This is less ideal as it doesn't have SQL types
            # We need to access the DataFrame directly
            # In the original app.py, this used a global variable, but we'll modify to avoid that
            
            # Try to get the DataFrame from the connection
            try:
                # This is a bit of a hack, but we can try to get the DataFrame from the connection
                # by executing a query that returns the first row
                df_info = connection.execute(f"SELECT * FROM {df_variable_name} LIMIT 1;").df()
                # Now get the full DataFrame
                df = connection.execute(f"SELECT * FROM {df_variable_name};").df()
                
                string_io = StringIO()
                df.info(buf=string_io)
                schema_string = f"Table (DataFrame Variable): {df_variable_name}\nPandas Info:\n{string_io.getvalue()}"
            except Exception as e3:
                print(f"Failed to get DataFrame info: {e3}")
                schema_string = f"Table (DataFrame Variable): {df_variable_name}\nSchema information unavailable."
        except Exception as e2:
            print(f"Introspection failed ({e2}). Creating basic schema string.")
            schema_string = f"Table (DataFrame Variable): {df_variable_name}\nSchema information unavailable."

    print("\n--- Schema for LLM ---")
    print(schema_string)
    return schema_string

def run_sql_query(sql_query: str, connection) -> str:
    """Executes the SQL query using DuckDB against the DataFrame variable and returns the result as a string."""
    print(f"\nExecuting SQL: {sql_query}")
    try:
        # DuckDB queries the DataFrame variable directly in the current scope
        result_df = connection.execute(sql_query).df()
        if result_df.empty:
            print("Query returned no results.")
            # Still return the empty DataFrame as a string to ensure LLM generates the answer
            print("Passing empty result to LLM for answer generation.")
        print("Query executed successfully.")
        return result_df.to_string()
    except Exception as e:
        print(f"Error executing SQL: {sql_query}\nError: {e}")
        # Provide specific feedback if the table name is wrong
        if "not found" in str(e) or "does not exist" in str(e):
             return f"Error executing SQL query: Could not find the data table. The generated SQL might be incorrect."
        return f"Error executing SQL query: {e}"

def sanitize_data_for_json(data):
    """
    Recursively sanitize data to ensure it's JSON-compliant.
    Replaces NaN, Infinity, and -Infinity with None.
    """
    import math
    
    if isinstance(data, dict):
        return {k: sanitize_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data_for_json(item) for item in data]
    elif isinstance(data, float):
        # Check for NaN, Infinity, -Infinity
        if math.isnan(data) or math.isinf(data):
            return None
    return data

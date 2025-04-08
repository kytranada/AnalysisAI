from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.data_utils import run_sql_query

def create_sql_chain(llm, schema_string: str, column_metadata_string: str):
    """Create a chain for SQL generation."""
    sql_prompt_template = PromptTemplate.from_template(
        """Given the following table schema:
{schema}

{column_metadata}

Write a SQL query compatible with DuckDB to answer the following question based on the schema and column metadata provided:
{question}

Instructions:
- The data is available in a table named '{table_name}'. Refer to this table in your query (e.g., SELECT column FROM {table_name}).
- Respond ONLY with the raw SQL query. Do not include explanations, markdown formatting (like ```sql), or any text before or after the query.
- Ensure column names are quoted correctly if they contain spaces or special characters (e.g., "Column Name"). The schema shows the exact names. Check the schema carefully.
- If the question asks for 'top N' or 'bottom N', include a LIMIT clause.
- If the question involves calculations (average, sum, difference), use appropriate SQL functions.

Examples:
- For counting by category: SELECT category_column, COUNT(*) FROM {table_name} GROUP BY category_column
- For finding average: SELECT AVG(numeric_column) FROM {table_name}
- For filtering and grouping: SELECT column_name, COUNT(*) FROM {table_name} WHERE condition GROUP BY column_name
- For counting total rows: SELECT COUNT(*) FROM {table_name}
- For counting based on a condition: SELECT COUNT(*) FROM {table_name} WHERE column_name = 'value'
"""
    )

    return (
        sql_prompt_template
        | llm
        | StrOutputParser()
    )

def create_answer_chain(llm):
    """Create a chain for answer generation."""
    answer_prompt_template = PromptTemplate.from_template(
        """Based on the following context (result of a SQL query), the original question, the SQL query that was executed, and the table schema, provide a concise, natural language answer.

Table Schema:
{schema}

Column Metadata:
{column_metadata}

SQL Query Executed:
{sql_query}

Context (Query Result):
{context}

Original Question: {question}

Answer the question based on the SQL query and its results. Make sure your answer accurately reflects the filters and conditions used in the SQL query.
"""
    )

    return (
        answer_prompt_template
        | llm
        | StrOutputParser()
    )

def create_full_qa_chain(llm, schema_string: str, column_metadata_string: str, connection, table_name: str = "user_df"):
    """Create the full QA chain that generates SQL, executes it, and generates an answer."""
    
    # SQL generation chain
    sql_generator = RunnablePassthrough.assign(
        schema=lambda _: schema_string,
        table_name=lambda _: table_name,
        column_metadata=lambda _: column_metadata_string
    ) | create_sql_chain(llm, schema_string, column_metadata_string) | RunnableLambda(
        lambda sql: sql.strip().replace('```sql', '').replace('```', '').strip()
    )
    
    # Full chain
    return (
        # 1. Generate SQL query
        RunnablePassthrough.assign(
            sql_query=lambda inputs: sql_generator.invoke(inputs)
        )
        # 2. Execute SQL query and get context
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda inputs: run_sql_query(inputs["sql_query"], connection)),
            schema=lambda _: schema_string,
            column_metadata=lambda _: column_metadata_string
        )
        # 3. Generate final answer
        | create_answer_chain(llm)
    )

class QueryResult:
    """Class to store the result of a query execution."""
    
    def __init__(self, answer: str, sql_query: str, execution_result: str):
        self.answer = answer
        self.sql_query = sql_query
        self.execution_result = execution_result
    
    def to_dict(self) -> Dict[str, str]:
        """Convert the result to a dictionary."""
        return {
            "answer": self.answer,
            "sql_query": self.sql_query,
            "execution_result": self.execution_result
        }

def execute_query(question: str, llm, connection, schema_string: str, 
                 column_metadata_string: str, table_name: str = "user_df") -> QueryResult:
    """
    Execute a natural language query and return the result.
    This is a more efficient version that avoids redundant SQL generation.
    """
    # Create the SQL generation chain
    sql_generator = RunnablePassthrough.assign(
        schema=lambda _: schema_string,
        table_name=lambda _: table_name,
        column_metadata=lambda _: column_metadata_string
    ) | create_sql_chain(llm, schema_string, column_metadata_string) | RunnableLambda(
        lambda sql: sql.strip().replace('```sql', '').replace('```', '').strip()
    )
    
    # Generate the SQL query
    sql_query = sql_generator.invoke({"question": question})
    
    # Execute the SQL query
    execution_result = run_sql_query(sql_query, connection)
    
    # Create the answer chain
    answer_chain = create_answer_chain(llm)
    
    # Generate the answer
    answer = answer_chain.invoke({
        "question": question,
        "sql_query": sql_query,
        "context": execution_result,
        "schema": schema_string,
        "column_metadata": column_metadata_string
    })
    
    # Return the result
    return QueryResult(answer, sql_query, execution_result)

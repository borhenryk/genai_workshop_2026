# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Part 1: Exploring Agent Building Blocks on Databricks
# MAGIC
# MAGIC Welcome to this hands-on workshop! In this notebook, we'll explore all the key components
# MAGIC for building AI agents on Databricks:
# MAGIC
# MAGIC 1. **Unity Catalog Functions** - Register and use custom Python functions as agent tools
# MAGIC 2. **Vector Search** - Create and query vector search indexes for RAG
# MAGIC 3. **MCP Servers** - Connect to managed and custom MCP servers
# MAGIC 4. **Genie Agent** - Query structured data with natural language
# MAGIC 5. **Lakebase Memory** - Short-term and long-term agent memory
# MAGIC
# MAGIC Each section is self-contained so you can run cells independently to understand each component.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Complete all `TODO`s throughout this notebook
# MAGIC - Have a Lakebase instance ready (optional for memory sections)
# MAGIC - Have a Genie Space created (optional for Genie section)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Install Dependencies

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-langchain[memory] databricks-agents mlflow-skinny[databricks] databricks-vectorsearch databricks-mcp langgraph uv mcp
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable MLflow Tracing
# MAGIC
# MAGIC **Important**: Run this cell first to enable automatic tracing. 
# MAGIC After each section, you'll see a "🔍 See the Trace" cell that runs a mini agent/chain 
# MAGIC so you can explore the MLflow trace in the UI.

# COMMAND ----------

# DBTITLE 1,Enable MLflow Autologging for Traces
import mlflow

# Enable automatic tracing for LangChain
mlflow.langchain.autolog()

print("✅ MLflow tracing enabled!")
print("📊 After running agent cells, click on the trace link in the output to explore the execution flow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Configuration values are loaded from `config.json`. Update that file for your environment.

# COMMAND ----------

# DBTITLE 1,Load Configuration from config.json
from helpers import load_config, print_config_summary

# Load configuration from config.json
config = load_config()

# Extract values for easy access
CATALOG = config["catalog"]
SCHEMA = config["schema"]
LLM_ENDPOINT_NAME = config["llm_endpoint_name"]
LAKEBASE_INSTANCE_NAME = config["lakebase_instance_name"]
GENIE_SPACE_ID = config["genie_space_id"]
EMBEDDING_ENDPOINT = config["embedding_endpoint"]
EMBEDDING_DIMS = config["embedding_dims"]
VECTOR_SEARCH_ENDPOINT = config["vector_search_endpoint"]

# Print configuration summary
print_config_summary(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 1: Unity Catalog Functions as Agent Tools
# MAGIC
# MAGIC Unity Catalog functions can be registered and used as tools for your agents.
# MAGIC This enables you to:
# MAGIC - Create reusable, governed tools
# MAGIC - Share tools across teams
# MAGIC - Version and manage tool definitions
# MAGIC
# MAGIC Let's create some example functions and see how to use them as agent tools.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Create a Simple UC Function

# COMMAND ----------

# DBTITLE 1,Create a Calculator Function in UC
# First, let's create a simple calculator function using SQL
spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.simple_calculator(
    operation STRING COMMENT 'The math operation: add, subtract, multiply, divide',
    a DOUBLE COMMENT 'First number',
    b DOUBLE COMMENT 'Second number'
)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'A simple calculator that performs basic math operations'
AS $$
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        return a / b if b != 0 else None
    else:
        return None
$$
""")

print(f"✅ Created function: {CATALOG}.{SCHEMA}.simple_calculator")

# COMMAND ----------

# DBTITLE 1,Test the UC Function directly
# Test the function with SQL
result = spark.sql(f"SELECT {CATALOG}.{SCHEMA}.simple_calculator('multiply', 7, 6) as result")
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Create a More Complex UC Function - Weather Lookup (Mock)

# COMMAND ----------

# DBTITLE 1,Create Weather Lookup Function
spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_weather(
    city STRING COMMENT 'The city name to get weather for'
)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Get the current weather for a city (mock data for demo)'
AS $$
    import json
    # Mock weather data for demo purposes
    weather_data = {{
        "new york": {{"temp": 72, "condition": "Sunny", "humidity": 45}},
        "london": {{"temp": 58, "condition": "Cloudy", "humidity": 78}},
        "tokyo": {{"temp": 68, "condition": "Partly Cloudy", "humidity": 62}},
        "paris": {{"temp": 65, "condition": "Rainy", "humidity": 85}},
        "sydney": {{"temp": 78, "condition": "Sunny", "humidity": 55}}
    }}
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return f"Weather in {{city}}: {{data['temp']}}°F, {{data['condition']}}, Humidity: {{data['humidity']}}%"
    return f"Weather data not available for {{city}}"
$$
""")

print(f"✅ Created function: {CATALOG}.{SCHEMA}.get_weather")

# COMMAND ----------

# DBTITLE 1,Test Weather Function
result = spark.sql(f"SELECT {CATALOG}.{SCHEMA}.get_weather('Tokyo') as weather")
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Use UC Functions as LangChain Tools

# COMMAND ----------

# DBTITLE 1,Load UC Functions as Tools
from databricks_langchain import UCFunctionToolkit, ChatDatabricks

# Create a toolkit with our UC functions
uc_toolkit = UCFunctionToolkit(
    function_names=[
        f"{CATALOG}.{SCHEMA}.simple_calculator",
        f"{CATALOG}.{SCHEMA}.get_weather"
    ]
)

# Get the tools
uc_tools = uc_toolkit.tools

print(f"Loaded {len(uc_tools)} UC function tools:")
for tool in uc_tools:
    print(f"  - {tool.name}: {tool.description[:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Use UC Tools with an LLM

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 See the Trace: UC Functions Agent
# MAGIC
# MAGIC Run this cell to create a simple agent that uses UC function tools.
# MAGIC **Check the MLflow trace** to see:
# MAGIC - LLM calls with tool binding
# MAGIC - Tool execution steps
# MAGIC - The full agent loop

# COMMAND ----------

# DBTITLE 1,🔍 TRACE: Simple UC Tools Agent
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from typing import Annotated, Sequence, TypedDict, Any, Optional
from langgraph.graph.message import add_messages
from databricks_langchain import ChatDatabricks

# Initialize the LLM
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# Bind the UC tools to the LLM
llm_with_tools = llm.bind_tools(uc_tools)

# Simple agent state
class UCAgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]

# Agent logic
def uc_should_continue(state):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

def uc_call_model(state):
    system = SystemMessage(content="You are a helpful assistant. Use tools to answer questions about weather and calculations.")
    response = llm_with_tools.invoke([system] + list(state["messages"]))
    return {"messages": [response]}

# Build mini agent
uc_workflow = StateGraph(UCAgentState)
uc_workflow.add_node("agent", RunnableLambda(uc_call_model))
uc_workflow.add_node("tools", ToolNode(uc_tools))
uc_workflow.set_entry_point("agent")
uc_workflow.add_conditional_edges("agent", uc_should_continue, {"tools": "tools", END: END})
uc_workflow.add_edge("tools", "agent")
uc_mini_agent = uc_workflow.compile()

# Run and see the trace!
print("🚀 Running UC Functions Agent - check the MLflow trace below!\n")
result = uc_mini_agent.invoke({
    "messages": [HumanMessage(content="What's 25 times 4? And what's the weather in Sydney?")]
})

print("Final response:")
print(result["messages"][-1].content)
print("\n👆 Click the trace link above to explore the execution flow!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 2: Vector Search for RAG
# MAGIC
# MAGIC Vector Search enables semantic search over your data, which is essential for
# MAGIC Retrieval-Augmented Generation (RAG) agents.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Create Sample Data for Vector Search

# COMMAND ----------

# DBTITLE 1,Create Sample Documents Table
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Sample documents about Databricks
documents = [
    (1, "Databricks is a unified analytics platform that combines data engineering, data science, and machine learning."),
    (2, "Unity Catalog provides unified governance for all data and AI assets across clouds."),
    (3, "Delta Lake is an open-source storage layer that brings reliability to data lakes."),
    (4, "MLflow is an open-source platform for managing the machine learning lifecycle."),
    (5, "Databricks SQL provides a native SQL experience for analytics and BI."),
    (6, "Mosaic AI Agent Framework helps you build, deploy, and monitor AI agents."),
    (7, "Lakebase is a transactional database for AI applications with short-term and long-term memory."),
    (8, "Vector Search enables similarity search over embeddings for RAG applications."),
    (9, "Genie allows users to ask natural language questions about their data."),
    (10, "Model Serving provides real-time inference for machine learning models.")
]

schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("content", StringType(), False)
])

# Create DataFrame and save to Delta table
df = spark.createDataFrame(documents, schema)
table_name = f"{CATALOG}.{SCHEMA}.workshop_documents"
df.write.mode("overwrite").saveAsTable(table_name)

print(f"✅ Created table: {table_name}")
display(spark.table(table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Create a Vector Search Endpoint (if not exists)

# COMMAND ----------

# DBTITLE 1,Create Vector Search Endpoint
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# VECTOR_SEARCH_ENDPOINT is loaded from config.json above

# Check if endpoint exists, create if not
try:
    endpoint = vsc.get_endpoint(VECTOR_SEARCH_ENDPOINT)
    print(f"✅ Vector Search endpoint exists: {VECTOR_SEARCH_ENDPOINT}")
except Exception as e:
    print(f"Creating Vector Search endpoint: {VECTOR_SEARCH_ENDPOINT}")
    vsc.create_endpoint(
        name=VECTOR_SEARCH_ENDPOINT,
        endpoint_type="STANDARD"
    )
    print(f"✅ Created Vector Search endpoint: {VECTOR_SEARCH_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Create a Vector Search Index

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.workshop_documents 
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print("✅ Enabled Change Data Feed on source table")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

INDEX_NAME = f"{CATALOG}.{SCHEMA}.workshop_documents_index"

# Create or sync the index
try:
    # Try to get existing index
    index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=INDEX_NAME)
    print(f"✅ Vector Search index exists: {INDEX_NAME}")
    # Sync the index to update with new data
    index.sync()
    print("Syncing index with latest data...")
except Exception as e:
    print(f"Creating Vector Search index: {INDEX_NAME}")
    # Create a Delta Sync index with managed embeddings
    index = vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        source_table_name=f"{CATALOG}.{SCHEMA}.workshop_documents",
        primary_key="id",
        pipeline_type="TRIGGERED",
        embedding_source_column="content",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT
    )
    print(f"✅ Created Vector Search index: {INDEX_NAME}")
    print("Note: Index creation takes a few minutes. Wait for it to be ONLINE before querying.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Query the Vector Search Index

# COMMAND ----------

# DBTITLE 1,Query Vector Search
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Get the index
index = vsc.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_NAME
)

# Check index status
index_status = index.describe().get("status", {}).get("ready", False)
print(f"Index ready: {index_status}")

# Query the index (once it's ONLINE)
if index_status:
    # Search for similar documents
    results = index.similarity_search(
        query_text="How do I manage governance for my data?",
        columns=["id", "content"],
        num_results=3
    )
    
    print("Search results for 'How do I manage governance for my data?':")
    for doc in results.get("result", {}).get("data_array", []):
        print(f"  - {doc}")
else:
    print("⏳ Index is still being created. Please wait and run this cell again.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Use Vector Search as an Agent Tool

# COMMAND ----------

# DBTITLE 1,Create VectorSearchRetrieverTool
from databricks_langchain import VectorSearchRetrieverTool

# Create a retriever tool (run when index is ONLINE)
vs_tool = VectorSearchRetrieverTool(
    index_name=INDEX_NAME,
    tool_name="databricks_docs_search",
    description="Search for information about Databricks products and features. Use this tool when users ask about Databricks, Unity Catalog, Delta Lake, MLflow, or other Databricks technologies.",
    num_results=3
)

print(f"✅ Created Vector Search retriever tool: {vs_tool.name}")

# Test the tool
result = vs_tool.invoke("What is Unity Catalog?")
print(f"\nSearch result:\n{result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 See the Trace: RAG Chain with Vector Search
# MAGIC
# MAGIC Run this cell to create a simple RAG chain.
# MAGIC **Check the MLflow trace** to see:
# MAGIC - Vector search retrieval
# MAGIC - Context augmentation
# MAGIC - LLM response generation

# COMMAND ----------

# DBTITLE 1,🔍 TRACE: Simple RAG Chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from helpers import format_docs, extract_text_from_response

# Create a simple RAG chain
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context from Databricks documentation.

Context:
{context}

Question: {question}

Answer:
""")

# Build the RAG chain with extraction as a step
rag_chain = (
    {
        "context": vs_tool | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(extract_text_from_response).with_config({"run_name": "ExtractText"})
)

# Run and see the trace!
print("🚀 Running RAG Chain - check the MLflow trace below!\n")

question = "What is Mosaic AI Agent Framework and how does it help with building agents?"
answer = rag_chain.invoke(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 3: MCP Servers
# MAGIC
# MAGIC Model Context Protocol (MCP) servers provide standardized tool interfaces.
# MAGIC Databricks supports both managed MCP servers (like Genie) and custom MCP servers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Connect to a Managed MCP Server (Genie)

# COMMAND ----------

# DBTITLE 1,Connect to Genie MCP Server
import asyncio
import nest_asyncio
nest_asyncio.apply()

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

w = WorkspaceClient()

# TODO: Update with your Genie Space ID
if GENIE_SPACE_ID:
    host = w.config.host
    genie_mcp_url = f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}"
    
    # Create MCP client
    mcp_client = DatabricksMCPClient(
        server_url=genie_mcp_url,
        workspace_client=w
    )
    
    # List available tools
    tools = mcp_client.list_tools()
    print(f"Available Genie MCP tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")
else:
    print("⏭️ Skipping Genie MCP - GENIE_SPACE_ID not configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Call a Genie MCP Tool

# COMMAND ----------

# DBTITLE 1,Call Genie MCP Tool
import time
import json
from helpers import poll_genie_response

if GENIE_SPACE_ID:
    # Call the Genie tool to ask a question
    response = mcp_client.call_tool(
        tool_name=tools[0].name,
        arguments={"query": "What tables are available?"}
    )
    
    # Extract initial response
    result_text = "".join([c.text for c in response.content if hasattr(c, 'text')])
    result = json.loads(result_text)
    
    # Poll for completion if still processing
    if result.get("status") in ["ASKING_AI", "EXECUTING_QUERY"]:
        print(f"⏳ Query processing... Status: {result.get('status')}")
        
        poll_result = poll_genie_response(
            mcp_client=mcp_client,
            tools=tools,
            conversation_id=result.get("conversationId"),
            message_id=result.get("messageId"),
            verbose=True
        )
        
        if poll_result["success"]:
            print(f"\n✅ Genie response:")
            for att in poll_result["content"]:
                print(att)
        else:
            print(f"❌ Query {poll_result['status']}: {poll_result['raw_result']}")
    else:
        print(f"Genie response:\n{result_text}")
else:
    print("⏭️ Skipping - GENIE_SPACE_ID not configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 See the Trace: MCP Tool Agent
# MAGIC
# MAGIC Run this cell to create an agent that uses MCP tools.
# MAGIC **Check the MLflow trace** to see:
# MAGIC - MCP tool discovery
# MAGIC - Tool invocation via MCP protocol
# MAGIC - Response handling

# COMMAND ----------

# DBTITLE 1,🔍 TRACE: MCP Tools Agent
if GENIE_SPACE_ID:
    import time
    import json
    from pydantic import BaseModel, create_model
    from langchain_core.tools import BaseTool
    from langchain_core.runnables import RunnableLambda
    from helpers import poll_genie_response, format_final_response
    
    # Create a LangChain tool wrapper for MCP with polling
    class GenieMCPTool(BaseTool):
        name: str = "genie_query"
        description: str = "Ask questions about your data using natural language"
        
        def _run(self, query: str) -> str:
            # Initial query
            response = mcp_client.call_tool(
                tool_name=tools[0].name,
                arguments={"query": query}
            )
            result_text = "".join([c.text for c in response.content if hasattr(c, 'text')])
            result = json.loads(result_text)
            
            # Poll if still processing
            if result.get("status") in ["ASKING_AI", "EXECUTING_QUERY"]:
                poll_result = poll_genie_response(
                    mcp_client=mcp_client,
                    tools=tools,
                    conversation_id=result.get("conversationId"),
                    message_id=result.get("messageId"),
                    verbose=False
                )
                
                if poll_result["success"]:
                    return "\n".join(poll_result["content"])
                else:
                    return f"Query {poll_result['status']}: {poll_result['raw_result']}"
            
            return result_text
    
    genie_mcp_tool = GenieMCPTool()
    llm_with_mcp = llm.bind_tools([genie_mcp_tool])
    
    # Simple MCP agent
    class MCPAgentState(TypedDict):
        messages: Annotated[Sequence, add_messages]
    
    def mcp_call_model(state):
        system = SystemMessage(content="You are a data analyst. Use the genie_query tool to answer data questions.")
        response = llm_with_mcp.invoke([system] + list(state["messages"]))
        return {"messages": [response]}
    
    def mcp_should_continue(state):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END
    
    def format_response_node(state):
        """Format the final response nicely - visible in trace"""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.content:
            content = last_message.content
            if isinstance(content, str):
                formatted_content = format_final_response(content, max_width=70)
                return {"messages": [AIMessage(content=formatted_content)]}
        return state
    
    mcp_workflow = StateGraph(MCPAgentState)
    mcp_workflow.add_node("agent", RunnableLambda(mcp_call_model))
    mcp_workflow.add_node("tools", ToolNode([genie_mcp_tool]))
    mcp_workflow.add_node("format_output", RunnableLambda(format_response_node).with_config({"run_name": "FormatFinalAnswer"}))
    mcp_workflow.set_entry_point("agent")
    mcp_workflow.add_conditional_edges("agent", mcp_should_continue, {"tools": "tools", END: "format_output"})
    mcp_workflow.add_edge("tools", "agent")
    mcp_workflow.add_edge("format_output", END)
    mcp_mini_agent = mcp_workflow.compile()
    
    print("🚀 Running MCP Agent - check the MLflow trace below!\n")
    result = mcp_mini_agent.invoke({
        "messages": [HumanMessage(content="What tables do you have access to?")]
    })
    print("Final response:")
    print(result["messages"][-1].content)
    print("\n👆 Click the trace link to see MCP tool invocation!")
else:
    print("⏭️ Skipping MCP trace demo - GENIE_SPACE_ID not configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 4: Genie Agent for Natural Language Data Queries
# MAGIC
# MAGIC Genie allows users to ask questions about their data in natural language.
# MAGIC It can generate SQL queries and provide insights from your tables.
# MAGIC
# MAGIC First, we'll create a sample sales dataset as a Delta table. You can then
# MAGIC create a **Genie Space** on top of this table in the Databricks UI:
# MAGIC 1. Go to the **Genie** tab in the left sidebar
# MAGIC 2. Click **New** and select the table created below
# MAGIC 3. Copy the Genie Space ID into your `config.json`

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Create a Sample Sales Table for Genie

# COMMAND ----------

# DBTITLE 1,Create Sample Sales Data
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from datetime import date

# Sample sales data
sales_data = [
    ("2025-01-05", "Laptop",       "Electronics", "Germany",  2, 1299.99, "Alice"),
    ("2025-01-08", "Headphones",   "Electronics", "Austria",  5,   89.99, "Bob"),
    ("2025-01-12", "Desk Chair",   "Furniture",   "Germany",  1,  449.00, "Alice"),
    ("2025-01-15", "Monitor",      "Electronics", "Poland",   3,  349.99, "Charlie"),
    ("2025-01-20", "Keyboard",     "Electronics", "Germany",  10,  69.99, "Bob"),
    ("2025-02-01", "Standing Desk","Furniture",   "Austria",  1,  799.00, "Alice"),
    ("2025-02-05", "Webcam",       "Electronics", "Poland",   4,   59.99, "Charlie"),
    ("2025-02-10", "Laptop",       "Electronics", "Germany",  1, 1299.99, "Bob"),
    ("2025-02-14", "Mouse",        "Electronics", "Austria",  8,   39.99, "Alice"),
    ("2025-02-20", "Bookshelf",    "Furniture",   "Germany",  2,  189.00, "Charlie"),
    ("2025-03-01", "Tablet",       "Electronics", "Poland",   3,  499.99, "Alice"),
    ("2025-03-05", "Desk Lamp",    "Furniture",   "Germany",  6,   45.00, "Bob"),
    ("2025-03-10", "Laptop",       "Electronics", "Austria",  2, 1299.99, "Charlie"),
    ("2025-03-15", "Headphones",   "Electronics", "Germany",  4,   89.99, "Alice"),
    ("2025-03-20", "Office Chair", "Furniture",   "Poland",   2,  549.00, "Bob"),
    ("2025-04-01", "Monitor",      "Electronics", "Germany",  2,  349.99, "Charlie"),
    ("2025-04-08", "Keyboard",     "Electronics", "Austria",  7,   69.99, "Alice"),
    ("2025-04-12", "Webcam",       "Electronics", "Germany",  3,   59.99, "Bob"),
    ("2025-04-18", "Standing Desk","Furniture",   "Poland",   1,  799.00, "Alice"),
    ("2025-04-25", "Tablet",       "Electronics", "Germany",  2,  499.99, "Charlie"),
]

schema = StructType([
    StructField("order_date", StringType(), False),
    StructField("product", StringType(), False),
    StructField("category", StringType(), False),
    StructField("country", StringType(), False),
    StructField("quantity", IntegerType(), False),
    StructField("unit_price", DoubleType(), False),
    StructField("sales_rep", StringType(), False),
])

df = spark.createDataFrame(sales_data, schema)

# Add computed columns
from pyspark.sql.functions import col, to_date, round as spark_round
df = (
    df
    .withColumn("order_date", to_date(col("order_date")))
    .withColumn("total_amount", spark_round(col("quantity") * col("unit_price"), 2))
)

# Save as Delta table
table_name = f"{CATALOG}.{SCHEMA}.workshop_sales"
df.write.mode("overwrite").saveAsTable(table_name)

print(f"✅ Created table: {table_name}")
print(f"   Rows: {df.count()}")
print(f"\n📌 Use this table to create a Genie Space in the Databricks UI,")
print(f"   then paste the Genie Space ID into config.json")
display(spark.table(table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Create a Genie Agent

# COMMAND ----------

# DBTITLE 1,Create Genie Agent
from databricks_langchain.genie import GenieAgent
import os

if GENIE_SPACE_ID:
    # For local development, we use the current user's credentials
    genie_agent = GenieAgent(
        genie_space_id=GENIE_SPACE_ID,
        genie_agent_name="DataAnalyst",
        description="A data analyst agent that can answer questions about your structured data using natural language."
    )
    
    print(f"✅ Created Genie Agent for space: {GENIE_SPACE_ID}")
else:
    print("⏭️ Skipping Genie Agent - GENIE_SPACE_ID not configured")

# COMMAND ----------

# DBTITLE 1,Query Genie Agent
if GENIE_SPACE_ID:
    # Ask a question to the Genie agent
    result = genie_agent.invoke({
        "messages": [{"role": "user", "content": "What data do you have access to?"}]
    })
    
    print("Genie Agent Response:")
    print(result["messages"][-1].content if result.get("messages") else "No response")
else:
    print("⏭️ Skipping - GENIE_SPACE_ID not configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 5: Lakebase Memory
# MAGIC
# MAGIC Lakebase provides durable memory storage for agents:
# MAGIC - **Short-term memory (Checkpoints)**: Conversation state within a thread
# MAGIC - **Long-term memory (Store)**: User preferences and facts across sessions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Setup Lakebase Tables

# COMMAND ----------

# DBTITLE 1,Setup Lakebase Checkpoints and Store
from databricks_langchain import CheckpointSaver, DatabricksStore

if LAKEBASE_INSTANCE_NAME and LAKEBASE_INSTANCE_NAME != "your-lakebase-instance":
    # Setup checkpoint tables for short-term memory
    with CheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME) as saver:
        saver.setup()
        print("✅ Checkpoint tables are ready for short-term memory.")
    
    # Setup store tables for long-term memory
    store = DatabricksStore(
        instance_name=LAKEBASE_INSTANCE_NAME,
        embedding_endpoint=EMBEDDING_ENDPOINT,
        embedding_dims=EMBEDDING_DIMS
    )
    store.setup()
    print("✅ Store tables are ready for long-term memory.")
else:
    print("⏭️ Skipping Lakebase setup - LAKEBASE_INSTANCE_NAME not configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Short-term Memory with CheckpointSaver

# COMMAND ----------

# DBTITLE 1,Short-term Memory Demo
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
import uuid

if LAKEBASE_INSTANCE_NAME and LAKEBASE_INSTANCE_NAME != "your-lakebase-instance":
    # Define a simple state
    class SimpleState(TypedDict):
        messages: Annotated[Sequence, add_messages]
    
    # Create a simple graph
    def echo_node(state):
        last_msg = state["messages"][-1].content if state["messages"] else ""
        return {"messages": [AIMessage(content=f"You said: {last_msg}")]}
    
    workflow = StateGraph(SimpleState)
    workflow.add_node("echo", echo_node)
    workflow.set_entry_point("echo")
    workflow.add_edge("echo", END)
    
    # Compile with checkpoint saver
    thread_id = str(uuid.uuid4())
    
    with CheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME) as saver:
        graph = workflow.compile(checkpointer=saver)
        
        # First message
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="Hello, I'm learning about agents!")]},
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"Response 1: {result1['messages'][-1].content}")
        
        # Second message in same thread - state is preserved
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="What did I just say?")]},
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"Response 2: {result2['messages'][-1].content}")
        
        print(f"\n✅ Conversation state saved to Lakebase with thread_id: {thread_id}")
else:
    print("⏭️ Skipping short-term memory demo - LAKEBASE_INSTANCE_NAME not configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Long-term Memory with DatabricksStore

# COMMAND ----------

# DBTITLE 1,Long-term Memory Demo
if LAKEBASE_INSTANCE_NAME and LAKEBASE_INSTANCE_NAME != "your-lakebase-instance":
    from databricks_langchain import DatabricksStore
    
    # Create store instance
    store = DatabricksStore(
        instance_name=LAKEBASE_INSTANCE_NAME,
        embedding_endpoint=EMBEDDING_ENDPOINT,
        embedding_dims=EMBEDDING_DIMS
    )
    
    # User ID for storing memories
    user_id = "workshop_user_1"
    namespace = ("user_memories", user_id)
    
    # Store some user preferences
    store.put(namespace, "preferences", {
        "favorite_language": "Python",
        "role": "Data Engineer",
        "experience": "5 years"
    })
    print("✅ Stored user preferences")
    
    store.put(namespace, "project_context", {
        "current_project": "Building AI agents workshop",
        "technology": "Databricks"
    })
    print("✅ Stored project context")
    
    # Retrieve memories using semantic search
    results = store.search(namespace, query="What programming language does the user prefer?", limit=3)
    
    print("\n🔍 Semantic search results for 'programming language preference':")
    for item in results:
        print(f"  - [{item.key}]: {item.value}")
else:
    print("⏭️ Skipping long-term memory demo - LAKEBASE_INSTANCE_NAME not configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 See the Trace: Agent with Memory Tools
# MAGIC
# MAGIC Run this cell to create an agent with memory capabilities.
# MAGIC **Check the MLflow trace** to see:
# MAGIC - Memory save operations
# MAGIC - Semantic search for memory retrieval
# MAGIC - Agent using remembered context

# COMMAND ----------

# DBTITLE 1,🔍 TRACE: Memory-Enhanced Agent
if LAKEBASE_INSTANCE_NAME and LAKEBASE_INSTANCE_NAME != "your-lakebase-instance":
    from langchain_core.tools import tool
    
    # Create memory tools
    @tool
    def remember_info(key: str, info: str) -> str:
        """Save information to long-term memory for later recall."""
        import json
        namespace = ("user_memories", "trace_demo_user")
        store.put(namespace, key, {"info": info})
        return f"Remembered: {key} = {info}"
    
    @tool
    def recall_info(query: str) -> str:
        """Search long-term memory for relevant information."""
        namespace = ("user_memories", "trace_demo_user")
        results = store.search(namespace, query=query, limit=3)
        if not results:
            return "No relevant memories found."
        return "\n".join([f"- {item.key}: {item.value}" for item in results])
    
    memory_tools = [remember_info, recall_info]
    llm_with_memory = llm.bind_tools(memory_tools)
    
    # Memory agent
    class MemoryAgentState(TypedDict):
        messages: Annotated[Sequence, add_messages]
    
    def memory_call_model(state):
        system = SystemMessage(content="You are an assistant with memory. Use remember_info to save important facts and recall_info to retrieve them.")
        response = llm_with_memory.invoke([system] + list(state["messages"]))
        return {"messages": [response]}
    
    def memory_should_continue(state):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END
    
    memory_workflow = StateGraph(MemoryAgentState)
    memory_workflow.add_node("agent", RunnableLambda(memory_call_model))
    memory_workflow.add_node("tools", ToolNode(memory_tools))
    memory_workflow.set_entry_point("agent")
    memory_workflow.add_conditional_edges("agent", memory_should_continue, {"tools": "tools", END: END})
    memory_workflow.add_edge("tools", "agent")
    memory_agent = memory_workflow.compile()
    
    print("🚀 Running Memory Agent - check the MLflow trace below!\n")
    
    # First, save something
    result1 = memory_agent.invoke({
        "messages": [HumanMessage(content="Please remember that (use the remember_info function) my favorite database is Delta Lake and I work on data pipelines.")]
    })
    print("Save response:", result1["messages"][-1].content[:200])
    
    # Then recall it
    result2 = memory_agent.invoke({
        "messages": [HumanMessage(content="What do you remember about my work (please use teh recall_info tool)?")]
    })
    print("\nRecall response:", result2["messages"][-1].content[:300])
    print("\n👆 Click the trace link to see memory save and semantic search!")
else:
    print("⏭️ Skipping memory trace demo - LAKEBASE_INSTANCE_NAME not configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 6: Complete Agent with Full Trace 🔍
# MAGIC
# MAGIC This is the culmination of everything we've learned. We'll create a complete agent
# MAGIC that combines multiple tools and shows the full execution trace.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Create a Complete Multi-Tool Agent

# COMMAND ----------

# DBTITLE 1,Build Complete Agent with Multiple Tools
from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from typing import Annotated, Sequence, TypedDict, Any, Optional
from langgraph.graph.message import add_messages

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]
    custom_inputs: Optional[dict[str, Any]]

# Get our UC tools
uc_toolkit = UCFunctionToolkit(
    function_names=[
        f"{CATALOG}.{SCHEMA}.simple_calculator",
        f"{CATALOG}.{SCHEMA}.get_weather"
    ]
)
tools = uc_toolkit.tools

# Create LLM with tools
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
You can:
- Perform calculations using the simple_calculator tool
- Get weather information using the get_weather tool

Always use the appropriate tool when the user asks for calculations or weather information."""

# Define agent logic
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"

def call_model(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", RunnableLambda(call_model))
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")

agent = workflow.compile()

print("✅ Created complete agent with UC function tools!")

# COMMAND ----------

agent.invoke({
        "messages": [HumanMessage(content="What tools do you have?")]
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Summary
# MAGIC
# MAGIC In this exploration notebook, you learned about:
# MAGIC
# MAGIC | Section | Component | Key Classes | Trace Shows |
# MAGIC |---------|-----------|-------------|-------------|
# MAGIC | 1 | **UC Functions** | `UCFunctionToolkit` | Tool binding & execution |
# MAGIC | 2 | **Vector Search** | `VectorSearchRetrieverTool` | Retrieval + RAG chain |
# MAGIC | 3 | **MCP Servers** | `DatabricksMCPClient` | MCP protocol calls |
# MAGIC | 4 | **Genie Agent** | `GenieAgent` | Data query processing |
# MAGIC | 5 | **Lakebase Memory** | `CheckpointSaver`, `DatabricksStore` | Memory operations |
# MAGIC | 6 | **Complete Agent** | `StateGraph`, `ToolNode` | Full agent loop |
# MAGIC
# MAGIC ## 🔍 What You Learned About MLflow Traces
# MAGIC
# MAGIC Each "See the Trace" cell demonstrated how MLflow automatically captures:
# MAGIC - **LLM Calls**: Input/output, tokens, latency
# MAGIC - **Tool Executions**: Arguments, results, timing
# MAGIC - **Chain/Graph Flow**: Node transitions, state changes
# MAGIC - **Retrieval**: Documents fetched, relevance scores
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to **Notebook 2** where we'll:
# MAGIC 1. Build a complete production-ready agent
# MAGIC 2. Wrap it with the `ResponsesAgent` interface
# MAGIC 3. Log it to MLflow
# MAGIC 4. Register it to Unity Catalog
# MAGIC 5. Deploy it to Model Serving
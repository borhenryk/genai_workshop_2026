# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Part 2: Build and Deploy a Production Agent
# MAGIC
# MAGIC In this notebook, we'll build a complete production-ready agent and deploy it to Databricks.
# MAGIC
# MAGIC We'll cover:
# MAGIC 1. **Define the Agent** - Create a LangGraph agent with tools and memory
# MAGIC 2. **Test Locally** - Verify the agent works before deployment
# MAGIC 3. **Log to MLflow** - Log the agent with resources for auth passthrough
# MAGIC 4. **Evaluate** - Run Mosaic AI Agent Evaluation (optional)
# MAGIC 5. **Register to Unity Catalog** - Version and govern your agent
# MAGIC 6. **Deploy to Model Serving** - Make the agent available via API
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Complete Notebook 1 (exploration) to understand the components
# MAGIC - Have a Lakebase instance ready (optional for memory features)
# MAGIC - Complete all `TODO`s throughout this notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Install Dependencies

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-langchain[memory] databricks-agents mlflow-skinny[databricks] uv langgraph
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Configuration values are loaded from `config.json`. Update that file for your environment.

# COMMAND ----------

# DBTITLE 1,Load Configuration from config.json
from helpers import load_config, print_config_summary, get_full_model_name

# Load configuration from config.json
config = load_config()

# Extract values for easy access
CATALOG = config["catalog"]
SCHEMA = config["schema"]
LLM_ENDPOINT_NAME = config["llm_endpoint_name"]
LAKEBASE_INSTANCE_NAME = config["lakebase_instance_name"]
EMBEDDING_ENDPOINT = config["embedding_endpoint"]
EMBEDDING_DIMS = config["embedding_dims"]
MODEL_NAME = config["model_name"]
VECTOR_SEARCH_ENDPOINT = config["vector_search_endpoint"]

# Print configuration summary
print_config_summary(config)
print(f"\nAgent will be registered as: {get_full_model_name(config)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Step 1: Define the Agent Code
# MAGIC
# MAGIC We'll write the agent code to a file using `%%writefile`. This is important because:
# MAGIC - MLflow logs agents "from code" for reproducibility
# MAGIC - The agent file becomes a self-contained, deployable artifact
# MAGIC
# MAGIC **Important**: `agent.py` must be **self-contained** (no imports from `helpers.py` or `config.json`).
# MAGIC When deployed to Model Serving, only `agent.py` is packaged. Make sure the config values
# MAGIC in `agent.py` match your `config.json`.
# MAGIC
# MAGIC Our agent will have:
# MAGIC - UC function tools (calculator, weather)
# MAGIC - Short-term memory with Lakebase checkpoints
# MAGIC - Long-term memory for user preferences
# MAGIC - ResponsesAgent interface for Databricks compatibility

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC """
# MAGIC Workshop Agent - A production-ready agent with tools and memory
# MAGIC
# MAGIC This agent demonstrates:
# MAGIC - Unity Catalog function tools
# MAGIC - Vector Search for RAG
# MAGIC - Short-term memory (conversation state) with Lakebase
# MAGIC - Long-term memory (user preferences) with semantic search
# MAGIC - ResponsesAgent interface for Databricks compatibility
# MAGIC """
# MAGIC import json
# MAGIC import logging
# MAGIC import os
# MAGIC import textwrap
# MAGIC import uuid
# MAGIC from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     UCFunctionToolkit,
# MAGIC     CheckpointSaver,
# MAGIC     DatabricksStore,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC )
# MAGIC from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import tool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC
# MAGIC logger = logging.getLogger(__name__)
# MAGIC logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
# MAGIC
# MAGIC ############################################
# MAGIC # Configuration
# MAGIC # NOTE: agent.py must be self-contained for
# MAGIC # MLflow logging and Model Serving deployment.
# MAGIC # Update these values to match config.json.
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"
# MAGIC LAKEBASE_INSTANCE_NAME = "lakebase_name"
# MAGIC EMBEDDING_ENDPOINT = "databricks-bge-large-en"
# MAGIC EMBEDDING_DIMS = 1024
# MAGIC CATALOG = "users_catalog"
# MAGIC SCHEMA = "users_schema"
# MAGIC VECTOR_SEARCH_ENDPOINT = "vs_endpoint"
# MAGIC VECTOR_SEARCH_INDEX = f"{CATALOG}.{SCHEMA}.workshop_documents_index_users_name"
# MAGIC
# MAGIC SYSTEM_PROMPT = """You are a helpful AI assistant for a workshop on building agents with Databricks.
# MAGIC
# MAGIC You have access to the following capabilities:
# MAGIC 1. Calculator - Perform mathematical calculations
# MAGIC 2. Weather - Get weather information for cities
# MAGIC 3. Document Search - Search the knowledge base for relevant information
# MAGIC 4. Memory - Remember and recall user preferences
# MAGIC
# MAGIC Guidelines:
# MAGIC - Be helpful, concise, and accurate
# MAGIC - Use the document search tool when users ask about Databricks, agents, or technical topics
# MAGIC - Use tools when appropriate to answer questions
# MAGIC - Remember important information users share about themselves
# MAGIC - If you recall information from memory, mention that you remembered it
# MAGIC """
# MAGIC
# MAGIC ############################################
# MAGIC # Define Tools
# MAGIC ############################################
# MAGIC tools = []
# MAGIC
# MAGIC # Add UC function tools
# MAGIC UC_TOOL_NAMES = [
# MAGIC     f"{CATALOG}.{SCHEMA}.simple_calculator",
# MAGIC     f"{CATALOG}.{SCHEMA}.get_weather"
# MAGIC ]
# MAGIC
# MAGIC try:
# MAGIC     uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC     tools.extend(uc_toolkit.tools)
# MAGIC     logger.info(f"Loaded {len(uc_toolkit.tools)} UC tools")
# MAGIC except Exception as e:
# MAGIC     logger.warning(f"Could not load UC tools: {e}")
# MAGIC
# MAGIC # Add Vector Search retriever tool
# MAGIC try:
# MAGIC     vs_tool = VectorSearchRetrieverTool(
# MAGIC         index_name=VECTOR_SEARCH_INDEX,
# MAGIC         num_results=3,
# MAGIC         columns=["id", "content"],
# MAGIC         tool_name="search_documentation",
# MAGIC         tool_description="Search the Databricks documentation knowledge base for information about agents, AI, Unity Catalog, and other Databricks topics. Use this when users ask technical questions."
# MAGIC     )
# MAGIC     tools.append(vs_tool)
# MAGIC     logger.info(f"Loaded Vector Search tool for index: {VECTOR_SEARCH_INDEX}")
# MAGIC except Exception as e:
# MAGIC     logger.warning(f"Could not load Vector Search tool: {e}")
# MAGIC
# MAGIC ############################################
# MAGIC # Response Formatter
# MAGIC ############################################
# MAGIC def format_final_response(content: str) -> str:
# MAGIC     """Format the final response for clean output."""
# MAGIC     if not content:
# MAGIC         return content
# MAGIC     if content.strip().startswith('[') or content.strip().startswith('{'):
# MAGIC         try:
# MAGIC             parsed = json.loads(content)
# MAGIC             if isinstance(parsed, list):
# MAGIC                 texts = []
# MAGIC                 for item in parsed:
# MAGIC                     if isinstance(item, dict) and item.get("type") == "text":
# MAGIC                         texts.append(item.get("text", ""))
# MAGIC                 if texts:
# MAGIC                     content = " ".join(texts)
# MAGIC         except json.JSONDecodeError:
# MAGIC             pass
# MAGIC     lines = content.strip().split('\n')
# MAGIC     formatted_lines = []
# MAGIC     for line in lines:
# MAGIC         if len(line) > 80:
# MAGIC             formatted_lines.append(textwrap.fill(line, width=80))
# MAGIC         else:
# MAGIC             formatted_lines.append(line)
# MAGIC     return '\n'.join(formatted_lines)
# MAGIC
# MAGIC ############################################
# MAGIC # Agent State
# MAGIC ############################################
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC     user_id: Optional[str]
# MAGIC     thread_id: Optional[str]
# MAGIC
# MAGIC ############################################
# MAGIC # ResponsesAgent Implementation
# MAGIC ############################################
# MAGIC class WorkshopAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     A production-ready agent with short-term and long-term memory.
# MAGIC     
# MAGIC     Features:
# MAGIC     - UC function tools for calculations and weather
# MAGIC     - Vector Search for RAG-based document retrieval
# MAGIC     - Short-term memory using Lakebase checkpoints
# MAGIC     - Long-term memory for user preferences with semantic search
# MAGIC     - ResponsesAgent interface for Databricks compatibility
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self):
# MAGIC         self.model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC         self.system_prompt = SYSTEM_PROMPT
# MAGIC         self._store = None
# MAGIC         self._memory_tools = None
# MAGIC
# MAGIC     @property
# MAGIC     def store(self):
# MAGIC         """Lazy initialization of DatabricksStore for long-term memory."""
# MAGIC         if self._store is None and LAKEBASE_INSTANCE_NAME and LAKEBASE_INSTANCE_NAME != "your-lakebase-instance":
# MAGIC             logger.info(f"Initializing DatabricksStore with instance: {LAKEBASE_INSTANCE_NAME}")
# MAGIC             self._store = DatabricksStore(
# MAGIC                 instance_name=LAKEBASE_INSTANCE_NAME,
# MAGIC                 embedding_endpoint=EMBEDDING_ENDPOINT,
# MAGIC                 embedding_dims=EMBEDDING_DIMS,
# MAGIC             )
# MAGIC             try:
# MAGIC                 self._store.setup()
# MAGIC             except Exception as e:
# MAGIC                 logger.warning(f"Could not setup store: {e}")
# MAGIC         return self._store
# MAGIC
# MAGIC     @property
# MAGIC     def memory_tools(self):
# MAGIC         """Lazy initialization of memory tools."""
# MAGIC         if self._memory_tools is None:
# MAGIC             self._memory_tools = self._create_memory_tools()
# MAGIC         return self._memory_tools
# MAGIC
# MAGIC     def _create_memory_tools(self):
# MAGIC         """Create tools for reading and writing long-term memory."""
# MAGIC         memory_tools = []
# MAGIC         
# MAGIC         if self.store is None:
# MAGIC             logger.info("Store not available, memory tools disabled")
# MAGIC             return memory_tools
# MAGIC
# MAGIC         @tool
# MAGIC         def get_user_memory(query: str, config: RunnableConfig) -> str:
# MAGIC             """Search for relevant information about the user from long-term memory.
# MAGIC             
# MAGIC             Use this to retrieve previously saved information about the user,
# MAGIC             such as preferences, facts they've shared, or personal details.
# MAGIC             
# MAGIC             Args:
# MAGIC                 query: What to search for in the user's memory
# MAGIC             """
# MAGIC             user_id = config.get("configurable", {}).get("user_id")
# MAGIC             if not user_id:
# MAGIC                 return "Memory not available - no user_id provided."
# MAGIC             
# MAGIC             namespace = ("user_memories", user_id.replace(".", "-"))
# MAGIC             results = self.store.search(namespace, query=query, limit=5)
# MAGIC             
# MAGIC             if not results:
# MAGIC                 return "No memories found for this user."
# MAGIC             
# MAGIC             memory_items = [f"- [{item.key}]: {json.dumps(item.value)}" for item in results]
# MAGIC             return f"Found {len(results)} relevant memories:\n" + "\n".join(memory_items)
# MAGIC
# MAGIC         @tool
# MAGIC         def save_user_memory(memory_key: str, memory_data_json: str, config: RunnableConfig) -> str:
# MAGIC             """Save information about the user to long-term memory.
# MAGIC             
# MAGIC             Use this to remember important information the user shares,
# MAGIC             such as preferences, facts, or personal details.
# MAGIC             
# MAGIC             Args:
# MAGIC                 memory_key: A descriptive key (e.g., "preferences", "favorite_color")
# MAGIC                 memory_data_json: JSON string with the information to remember
# MAGIC             """
# MAGIC             user_id = config.get("configurable", {}).get("user_id")
# MAGIC             if not user_id:
# MAGIC                 return "Cannot save memory - no user_id provided."
# MAGIC             
# MAGIC             namespace = ("user_memories", user_id.replace(".", "-"))
# MAGIC             
# MAGIC             try:
# MAGIC                 memory_data = json.loads(memory_data_json)
# MAGIC                 if not isinstance(memory_data, dict):
# MAGIC                     return f"Failed: memory_data must be a JSON object, not {type(memory_data).__name__}"
# MAGIC                 self.store.put(namespace, memory_key, memory_data)
# MAGIC                 return f"Successfully saved memory with key '{memory_key}'."
# MAGIC             except json.JSONDecodeError as e:
# MAGIC                 return f"Failed: Invalid JSON format - {str(e)}"
# MAGIC
# MAGIC         memory_tools = [get_user_memory, save_user_memory]
# MAGIC         return memory_tools
# MAGIC
# MAGIC     @property
# MAGIC     def all_tools(self):
# MAGIC         """Get all available tools (UC functions + Vector Search + memory tools)."""
# MAGIC         return tools + self.memory_tools
# MAGIC
# MAGIC     @property
# MAGIC     def model_with_tools(self):
# MAGIC         """LLM with all tools bound."""
# MAGIC         all_tools = self.all_tools
# MAGIC         return self.model.bind_tools(all_tools) if all_tools else self.model
# MAGIC
# MAGIC     def _create_graph(self, checkpointer=None):
# MAGIC         """Create the LangGraph workflow."""
# MAGIC         
# MAGIC         def should_continue(state: AgentState):
# MAGIC             messages = state["messages"]
# MAGIC             last_message = messages[-1]
# MAGIC             if isinstance(last_message, AIMessage) and last_message.tool_calls:
# MAGIC                 return "continue"
# MAGIC             return "format"
# MAGIC
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": self.system_prompt}] + state["messages"]
# MAGIC         )
# MAGIC         model_runnable = preprocessor | self.model_with_tools
# MAGIC
# MAGIC         def call_model(state: AgentState, config: RunnableConfig):
# MAGIC             response = model_runnable.invoke(state, config)
# MAGIC             return {"messages": [response]}
# MAGIC
# MAGIC         def format_response(state: AgentState, config: RunnableConfig):
# MAGIC             """Format the final response - appears as 'FormatResponse' in trace."""
# MAGIC             last_message = state["messages"][-1]
# MAGIC             if isinstance(last_message, AIMessage) and last_message.content:
# MAGIC                 content = last_message.content
# MAGIC                 if isinstance(content, str):
# MAGIC                     formatted_content = format_final_response(content)
# MAGIC                     return {"messages": [AIMessage(content=formatted_content)]}
# MAGIC             return {"messages": []}
# MAGIC
# MAGIC         workflow = StateGraph(AgentState)
# MAGIC         workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC         workflow.add_node("format_response", RunnableLambda(format_response).with_config({"run_name": "FormatFinalAnswer"}))
# MAGIC
# MAGIC         all_tools = self.all_tools
# MAGIC         if all_tools:
# MAGIC             workflow.add_node("tools", ToolNode(all_tools))
# MAGIC             workflow.add_conditional_edges(
# MAGIC                 "agent", should_continue, {"continue": "tools", "format": "format_response"}
# MAGIC             )
# MAGIC             workflow.add_edge("tools", "agent")
# MAGIC         else:
# MAGIC             workflow.add_edge("agent", "format_response")
# MAGIC
# MAGIC         workflow.add_edge("format_response", END)
# MAGIC         workflow.set_entry_point("agent")
# MAGIC         return workflow.compile(checkpointer=checkpointer)
# MAGIC
# MAGIC     def _get_thread_id(self, request: ResponsesAgentRequest) -> str:
# MAGIC         """Get thread_id from request or create a new one."""
# MAGIC         ci = dict(request.custom_inputs or {})
# MAGIC         
# MAGIC         if "thread_id" in ci:
# MAGIC             return ci["thread_id"]
# MAGIC         
# MAGIC         if request.context and getattr(request.context, "conversation_id", None):
# MAGIC             return request.context.conversation_id
# MAGIC         
# MAGIC         return str(uuid.uuid4())
# MAGIC
# MAGIC     def _get_user_id(self, request: ResponsesAgentRequest) -> Optional[str]:
# MAGIC         """Get user_id from request context."""
# MAGIC         if request.context and getattr(request.context, "user_id", None):
# MAGIC             return request.context.user_id
# MAGIC         return dict(request.custom_inputs or {}).get("user_id")
# MAGIC
# MAGIC     def _set_trace_session(self, request: ResponsesAgentRequest, thread_id: str):
# MAGIC         """Set session metadata on the MLflow trace for conversation grouping."""
# MAGIC         session_id = thread_id
# MAGIC         if request.custom_inputs and "session_id" in request.custom_inputs:
# MAGIC             session_id = request.custom_inputs["session_id"]
# MAGIC         elif request.context and getattr(request.context, "conversation_id", None):
# MAGIC             session_id = request.context.conversation_id
# MAGIC         
# MAGIC         try:
# MAGIC             mlflow.update_current_trace(
# MAGIC                 metadata={"mlflow.trace.session": session_id}
# MAGIC             )
# MAGIC         except Exception:
# MAGIC             pass  # Tracing may not be active
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         """Non-streaming prediction."""
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         """Streaming prediction with memory support."""
# MAGIC         thread_id = self._get_thread_id(request)
# MAGIC         user_id = self._get_user_id(request)
# MAGIC         
# MAGIC         # Set trace session for conversation grouping in MLflow UI
# MAGIC         self._set_trace_session(request, thread_id)
# MAGIC         
# MAGIC         # Update custom_inputs with resolved IDs
# MAGIC         ci = dict(request.custom_inputs or {})
# MAGIC         ci["thread_id"] = thread_id
# MAGIC         if user_id:
# MAGIC             ci["user_id"] = user_id
# MAGIC         request.custom_inputs = ci
# MAGIC
# MAGIC         # Convert incoming messages using standard Responses API conversion
# MAGIC         cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC         
# MAGIC         # Build config
# MAGIC         run_config = {"configurable": {"thread_id": thread_id}}
# MAGIC         if user_id:
# MAGIC             run_config["configurable"]["user_id"] = user_id
# MAGIC
# MAGIC         # Execute with optional checkpointing
# MAGIC         if LAKEBASE_INSTANCE_NAME and LAKEBASE_INSTANCE_NAME != "your-lakebase-instance":
# MAGIC             with CheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME) as checkpointer:
# MAGIC                 graph = self._create_graph(checkpointer=checkpointer)
# MAGIC                 yield from self._stream_graph(graph, cc_msgs, run_config)
# MAGIC         else:
# MAGIC             graph = self._create_graph()
# MAGIC             yield from self._stream_graph(graph, cc_msgs, run_config)
# MAGIC
# MAGIC     def _stream_graph(self, graph, messages, config):
# MAGIC         """Stream events from the graph execution.
# MAGIC         
# MAGIC         Uses 'updates' stream mode so we can filter by node name.
# MAGIC         - 'agent' node: only yield tool-calling messages (function_call items)
# MAGIC         - 'tools' node: yield tool results (function_call_output items)
# MAGIC         - 'format_response' node: yield the cleaned final text
# MAGIC         
# MAGIC         This prevents duplicate output: the raw AI message from the agent
# MAGIC         node is NOT yielded; only the formatted version from format_response is.
# MAGIC         """
# MAGIC         for event in graph.stream(
# MAGIC             {"messages": messages},
# MAGIC             config,
# MAGIC             stream_mode="updates"
# MAGIC         ):
# MAGIC             for node_name, node_data in event.items():
# MAGIC                 msgs = node_data.get("messages", [])
# MAGIC                 if not msgs:
# MAGIC                     continue
# MAGIC                 if node_name == "agent":
# MAGIC                     # Only yield tool-calling messages from the agent node.
# MAGIC                     # The final text response will come from format_response instead.
# MAGIC                     tool_msgs = [m for m in msgs if isinstance(m, AIMessage) and m.tool_calls]
# MAGIC                     if tool_msgs:
# MAGIC                         yield from output_to_responses_items_stream(tool_msgs)
# MAGIC                 else:
# MAGIC                     # Yield everything from 'tools' and 'format_response' nodes
# MAGIC                     yield from output_to_responses_items_stream(msgs)
# MAGIC
# MAGIC
# MAGIC # ----- Export model -----
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = WorkshopAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Step 2: Test the Agent Locally
# MAGIC
# MAGIC Before deploying, we should test the agent to make sure it works correctly.
# MAGIC
# MAGIC **🔍 MLflow Tracing**: Each test cell will generate a trace you can explore in the UI.
# MAGIC Click the trace link in the output to see the full execution flow!

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Reload Configuration after Restart
from helpers import load_config, print_config_summary, get_full_model_name

# Reload config since restartPython() clears all variables
config = load_config()
CATALOG = config["catalog"]
SCHEMA = config["schema"]
LLM_ENDPOINT_NAME = config["llm_endpoint_name"]
LAKEBASE_INSTANCE_NAME = config["lakebase_instance_name"]
EMBEDDING_ENDPOINT = config["embedding_endpoint"]
EMBEDDING_DIMS = config["embedding_dims"]
MODEL_NAME = config["model_name"]
VECTOR_SEARCH_ENDPOINT = config["vector_search_endpoint"]

print("✅ Configuration reloaded after kernel restart")
print_config_summary(config)

# COMMAND ----------

# DBTITLE 1,🔍 TRACE: Test Basic Agent Response
from agent import AGENT

print("🚀 Testing basic response - check the MLflow trace!\n")

# Test basic conversation
result = AGENT.predict({
    "input": [{"role": "user", "content": "Hello! What can you help me with?"}]
})

print("Agent response:")
print(result.model_dump(exclude_none=True))
print("\n👆 Click the trace link above to see the agent's reasoning!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Step 3: Log the Agent to MLflow
# MAGIC
# MAGIC Now we'll log the agent as an MLflow model. Key points:
# MAGIC - Specify **resources** for automatic authentication passthrough
# MAGIC - Use **models from code** pattern for reproducibility
# MAGIC - Include proper **pip requirements**

# COMMAND ----------

# DBTITLE 1,Define Resources and Log Model
import mlflow
from agent import tools, LLM_ENDPOINT_NAME, LAKEBASE_INSTANCE_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import (
    DatabricksFunction, 
    DatabricksServingEndpoint, 
    DatabricksLakebase
)
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
from pkg_resources import get_distribution

# Define resources for automatic authentication passthrough
resources = [
    DatabricksServingEndpoint(LLM_ENDPOINT_NAME),
    DatabricksServingEndpoint(EMBEDDING_ENDPOINT),  # For embeddings
]

# Add Lakebase if configured
if LAKEBASE_INSTANCE_NAME and LAKEBASE_INSTANCE_NAME != "your-lakebase-instance":
    resources.append(DatabricksLakebase(database_instance_name=LAKEBASE_INSTANCE_NAME))

# Add UC function resources
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))
    elif hasattr(tool, 'name') and '.' in str(tool.name):
        # Try to extract function name for UC tools
        resources.append(DatabricksFunction(function_name=tool.name))

print(f"Resources for auth passthrough ({len(resources)}):")
for r in resources:
    print(f"  - {type(r).__name__}: {r}")

# COMMAND ----------

# DBTITLE 1,Log the Model
# Input example for logging
input_example = {
    "input": [
        {
            "role": "user",
            "content": "What is an AI agent and how can I build one on Databricks?"
        }
    ],
    "custom_inputs": {"thread_id": "example-thread-123"},
}

# Log the model
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            f"databricks-langchain[memory]=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"mlflow=={get_distribution('mlflow').version}",
        ]
    )

print(f"✅ Logged agent to MLflow run: {logged_agent_info.run_id}")
print(f"   Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Step 4: Agent Evaluation (Optional)
# MAGIC
# MAGIC Evaluate the agent's responses using Mosaic AI Agent Evaluation.

# COMMAND ----------

# DBTITLE 1,Evaluate Agent
from mlflow.genai.scorers import RelevanceToQuery, Safety
from agent import AGENT

# Define evaluation dataset
eval_dataset = [
    {
        "inputs": {"input": [{"role": "user", "content": "Calculate 15 times 8"}]},
        "expected_response": "120",
    },
    {
        "inputs": {"input": [{"role": "user", "content": "What's the weather in London?"}]},
        "expected_response": "Weather information for London",
    },
    {
        "inputs": {"input": [{"role": "user", "content": "Hello, how are you?"}]},
        "expected_response": "A friendly greeting response",
    },
]

# Run evaluation
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=[RelevanceToQuery(), Safety()],
)

print("✅ Evaluation complete - check MLflow UI for detailed results")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Step 5: Register to Unity Catalog
# MAGIC
# MAGIC Register the agent model to Unity Catalog for versioning and governance.

# COMMAND ----------

# DBTITLE 1,Register Model to Unity Catalog
import mlflow
from helpers import get_full_model_name

mlflow.set_registry_uri("databricks-uc")

UC_MODEL_NAME = get_full_model_name(config)

print(f"Registering model to: {UC_MODEL_NAME}")

# Register the model
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name=UC_MODEL_NAME
)

print(f"✅ Registered model version: {uc_registered_model_info.version}")
print(f"   Model name: {UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Step 6: Deploy to Model Serving
# MAGIC
# MAGIC Deploy the agent to a Model Serving endpoint.

# COMMAND ----------

# DBTITLE 1,Deploy Agent
from databricks import agents
from helpers import wait_for_endpoint_ready, get_full_model_name
from databricks.sdk import WorkspaceClient

UC_MODEL_NAME = get_full_model_name(config)

# Deploy the agent
print("🚀 Deploying agent...")
try:
    deployment = agents.deploy(
        UC_MODEL_NAME,
        uc_registered_model_info.version,
        scale_to_zero_enabled=True,
        deploy_feedback_model=True
    )
except Exception as e:
    if "Duplicate key" in str(e):
        # Endpoint already exists with old tags - deploy without tags
        print("⚠️ Endpoint exists with conflicting tags, redeploying...")
        deployment = agents.deploy(
            UC_MODEL_NAME,
            uc_registered_model_info.version,
            deploy_feedback_model=True
        )
    else:
        raise

# Get the actual endpoint name from the deployment object
endpoint_name = deployment.endpoint_name
print(f"✅ Deployment initiated!")
print(f"   Endpoint name: {endpoint_name}")

# Wait for endpoint to be ready using helper function
result = wait_for_endpoint_ready(
    endpoint_name=endpoint_name,
    max_wait_minutes=40,
    poll_interval_seconds=120,
    verbose=True
)

if result["success"]:
    print(f"\n🎉 Deployment successful!")
else:
    print(f"\n⚠️ {result['message']}")

w = WorkspaceClient()
print(f"\n📊 Check status in Model Serving UI:")
print(f"   https://{w.config.host}/ml/endpoints/{endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Next Steps
# MAGIC
# MAGIC Congratulations! You've built and deployed a production agent on Databricks! 🎉
# MAGIC
# MAGIC ## What you've accomplished:
# MAGIC
# MAGIC | Step | What we did |
# MAGIC |------|------------|
# MAGIC | 1 | Defined agent with UC tools and Lakebase memory |
# MAGIC | 2 | Wrapped with ResponsesAgent for Databricks compatibility |
# MAGIC | 3 | Tested locally with various scenarios |
# MAGIC | 4 | Logged to MLflow with auth passthrough resources |
# MAGIC | 5 | Registered to Unity Catalog |
# MAGIC | 6 | Deployed to Model Serving |
# MAGIC
# MAGIC ## After deployment:
# MAGIC
# MAGIC 1. **Test in AI Playground**: Chat with your agent in the Databricks AI Playground
# MAGIC 2. **Share for Feedback**: Share with SMEs using the Review App
# MAGIC 3. **Query via API**: Call the endpoint programmatically
# MAGIC 4. **Monitor**: Track usage and performance in the Model Serving UI
# MAGIC
# MAGIC ## Query your Lakebase for conversation history:
# MAGIC
# MAGIC ```sql
# MAGIC -- View recent checkpoints (short-term memory)
# MAGIC SELECT
# MAGIC     c.*,
# MAGIC     (c.checkpoint::json->>'ts')::timestamptz AS ts
# MAGIC FROM checkpoints c
# MAGIC ORDER BY ts DESC
# MAGIC LIMIT 10;
# MAGIC
# MAGIC -- View user memories (long-term memory)
# MAGIC SELECT *
# MAGIC FROM public.store
# MAGIC ORDER BY updated_at DESC
# MAGIC LIMIT 50;
# MAGIC ```

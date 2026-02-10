# Databricks Agent Workshop

Hands-on workshop for building and deploying AI agents on Databricks using LangGraph, MLflow, and the Databricks AI stack.

## Workshop Notebooks

### `01_workshop_exploration.py`
Explore the building blocks for agents on Databricks — each section is self-contained:

- **Unity Catalog Functions** – Register custom Python functions as agent tools
- **Vector Search** – Create and query vector indexes for RAG
- **MCP Servers** – Connect to managed and custom MCP servers (incl. Genie)
- **Genie Agent** – Query structured data with natural language
- **Lakebase Memory** – Short-term (checkpoints) and long-term (semantic store) memory

### `02_workshop_build_deploy_agent.py`
Build a production-ready agent and deploy it end-to-end:

1. Define a LangGraph agent with UC tools, Vector Search, and Lakebase memory
2. Test locally with MLflow tracing
3. Log to MLflow with auth-passthrough resources
4. Evaluate with Mosaic AI Agent Evaluation
5. Register to Unity Catalog
6. Deploy to Model Serving

## Setup

### 1. Update `config.json`

All configurable parameters live in `config.json`. Update these for your environment:

| Parameter | Description |
|-----------|-------------|
| `catalog` | Your Unity Catalog catalog name |
| `schema` | Your Unity Catalog schema name |
| `llm_endpoint_name` | Model Serving endpoint for the LLM |
| `embedding_endpoint` | Embedding model endpoint |
| `embedding_dims` | Embedding dimensions (1024 for `databricks-bge-large-en`) |
| `vector_search_endpoint` | Vector Search endpoint name |
| `lakebase_instance_name` | Lakebase instance for agent memory |
| `genie_space_id` | Genie Space ID (for notebook 1 only) |
| `model_name` | Name for the registered UC model |

### 2. Update `agent.py` config block (important!)

The `agent.py` file is written via `%%writefile` in notebook 2 and deployed as a **self-contained artifact** to Model Serving. It does **not** read from `config.json` at serving time.

You must manually update the configuration block at the top of the `%%writefile agent.py` cell to match your `config.json` values:

```python
############################################
# Configuration
# NOTE: agent.py must be self-contained for
# MLflow logging and Model Serving deployment.
# Update these values to match config.json.
############################################
LLM_ENDPOINT_NAME = "your-llm-endpoint"        # ← update
LAKEBASE_INSTANCE_NAME = "your-lakebase"        # ← update
EMBEDDING_ENDPOINT = "databricks-bge-large-en"  # ← update
EMBEDDING_DIMS = 1024
CATALOG = "your_catalog"                        # ← update
SCHEMA = "your_schema"                          # ← update
VECTOR_SEARCH_ENDPOINT = "your-vs-endpoint"     # ← update
```

### 3. Lakebase

A single Lakebase instance can be shared across all workshop participants — each conversation is isolated by `thread_id` and `user_id`.

## Files

| File | Purpose |
|------|---------|
| `01_workshop_exploration.py` | Notebook 1 — explore agent building blocks |
| `02_workshop_build_deploy_agent.py` | Notebook 2 — build and deploy the agent |
| `config.json` | Shared configuration (used by notebooks, **not** by `agent.py`) |
| `helpers.py` | Utility functions: config loading, formatting, deploy polling |

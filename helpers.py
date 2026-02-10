"""
Workshop Helpers - Utility functions for the agent workshop notebooks

This module contains:
- Configuration loading
- Response formatting functions
- Deployment waiting/polling functions
- Other utility functions
"""

import json
import os
import textwrap
import time
from typing import Any, Optional

# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from config.json file.
    
    Args:
        config_path: Optional path to config file. Defaults to config.json in same directory.
    
    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        # Try to find config.json relative to this file or current working directory
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "config.json"),
            "config.json",
            "/Workspace/Users/" + os.environ.get("USER", "") + "/config.json",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError(
            "config.json not found. Please create it with your configuration values."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_config_value(key: str, default: Any = None, config: dict = None) -> Any:
    """
    Get a specific configuration value.
    
    Args:
        key: Configuration key to retrieve
        default: Default value if key not found
        config: Optional pre-loaded config dict
    
    Returns:
        Configuration value
    """
    if config is None:
        config = load_config()
    return config.get(key, default)


# ============================================================================
# Response Formatting Functions
# ============================================================================

def extract_text_from_response(response: str) -> str:
    """
    Extract clean text from LLM response (handles JSON structured output).
    
    Some models return responses as JSON arrays with type/text objects.
    This function extracts the text content from such responses.
    
    Args:
        response: Raw response string from LLM
    
    Returns:
        Clean text content
    """
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                texts = []
                for item in parsed:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                return " ".join(texts) if texts else response
            return response
        except json.JSONDecodeError:
            return response
    return str(response)


def format_final_response(content: str, max_width: int = 80) -> str:
    """
    Format the final response for clean, readable output.
    
    Handles JSON structured responses and wraps long lines.
    
    Args:
        content: Raw content to format
        max_width: Maximum line width for wrapping
    
    Returns:
        Formatted content string
    """
    if not content:
        return content
    
    # Handle JSON structured responses (from some models)
    if content.strip().startswith('[') or content.strip().startswith('{'):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                texts = []
                for item in parsed:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                if texts:
                    content = " ".join(texts)
        except json.JSONDecodeError:
            pass
    
    # Clean up and wrap text
    lines = content.strip().split('\n')
    formatted_lines = []
    for line in lines:
        if len(line) > max_width:
            wrapped = textwrap.fill(line, width=max_width)
            formatted_lines.append(wrapped)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def format_docs(docs: Any) -> str:
    """
    Format documents for RAG context.
    
    Args:
        docs: Documents to format (string or list of strings)
    
    Returns:
        Formatted document string
    """
    if isinstance(docs, str):
        return docs
    if isinstance(docs, list):
        return "\n".join(str(doc) for doc in docs)
    return str(docs)


# ============================================================================
# Deployment Waiting Functions
# ============================================================================

def wait_for_endpoint_ready(
    endpoint_name: str,
    max_wait_minutes: int = 40,
    poll_interval_seconds: int = 120,
    verbose: bool = True
) -> dict:
    """
    Wait for a Model Serving endpoint to become ready.
    
    Args:
        endpoint_name: Name of the serving endpoint
        max_wait_minutes: Maximum time to wait in minutes
        poll_interval_seconds: Time between status checks in seconds
        verbose: Whether to print status updates
    
    Returns:
        Dictionary with status information:
        - success: Whether endpoint is ready
        - state: Final state of endpoint
        - message: Status message
        - endpoint: Endpoint object (if successful)
    """
    from databricks.sdk import WorkspaceClient
    
    w = WorkspaceClient()
    max_attempts = (max_wait_minutes * 60) // poll_interval_seconds
    
    if verbose:
        print(f"⏳ Waiting for endpoint '{endpoint_name}' to be ready...")
        print(f"   (max wait: {max_wait_minutes} minutes, polling every {poll_interval_seconds}s)\n")
    
    for attempt in range(max_attempts):
        try:
            endpoint = w.serving_endpoints.get(endpoint_name)
            state = endpoint.state
            
            # Compare as string to handle both enum and string values
            ready_str = str(state.ready) if state.ready else ""
            config_str = str(state.config_update) if state.config_update else ""
            
            if "READY" in ready_str:
                if verbose:
                    print(f"\n✅ Endpoint is READY!")
                return {
                    "success": True,
                    "state": "READY",
                    "message": "Endpoint is ready and serving",
                    "endpoint": endpoint
                }
            elif "FAILED" in config_str:
                if verbose:
                    print(f"\n❌ Deployment failed!")
                    print(f"   State: {state}")
                return {
                    "success": False,
                    "state": "UPDATE_FAILED",
                    "message": f"Deployment failed: {state}",
                    "endpoint": endpoint
                }
            else:
                status = ready_str or config_str or "UNKNOWN"
                if verbose:
                    print(f"   [{attempt + 1}/{max_attempts}] Status: {status}")
                time.sleep(poll_interval_seconds)
                
        except Exception as e:
            if verbose:
                print(f"   [{attempt + 1}/{max_attempts}] Endpoint not found yet: {e}")
            time.sleep(poll_interval_seconds)
    
    if verbose:
        print(f"\n⏰ Timeout after {max_wait_minutes} minutes")
    
    return {
        "success": False,
        "state": "TIMEOUT",
        "message": f"Timeout waiting for endpoint after {max_wait_minutes} minutes",
        "endpoint": None
    }


def wait_for_vector_search_index_ready(
    endpoint_name: str,
    index_name: str,
    max_wait_minutes: int = 10,
    poll_interval_seconds: int = 30,
    verbose: bool = True
) -> dict:
    """
    Wait for a Vector Search index to become ready.
    
    Args:
        endpoint_name: Name of the vector search endpoint
        index_name: Full name of the index (catalog.schema.index_name)
        max_wait_minutes: Maximum time to wait in minutes
        poll_interval_seconds: Time between status checks in seconds
        verbose: Whether to print status updates
    
    Returns:
        Dictionary with status information:
        - success: Whether index is ready
        - message: Status message
        - index: Index object (if successful)
    """
    from databricks.vector_search.client import VectorSearchClient
    
    vsc = VectorSearchClient()
    max_attempts = (max_wait_minutes * 60) // poll_interval_seconds
    
    if verbose:
        print(f"⏳ Waiting for index '{index_name}' to be ready...")
    
    for attempt in range(max_attempts):
        try:
            index = vsc.get_index(
                endpoint_name=endpoint_name,
                index_name=index_name
            )
            status = index.describe().get("status", {})
            is_ready = status.get("ready", False)
            
            if is_ready:
                if verbose:
                    print(f"\n✅ Index is READY!")
                return {
                    "success": True,
                    "message": "Index is ready for queries",
                    "index": index
                }
            else:
                if verbose:
                    print(f"   [{attempt + 1}/{max_attempts}] Index not ready yet...")
                time.sleep(poll_interval_seconds)
                
        except Exception as e:
            if verbose:
                print(f"   [{attempt + 1}/{max_attempts}] Error checking index: {e}")
            time.sleep(poll_interval_seconds)
    
    if verbose:
        print(f"\n⏰ Timeout after {max_wait_minutes} minutes")
    
    return {
        "success": False,
        "message": f"Timeout waiting for index after {max_wait_minutes} minutes",
        "index": None
    }


# ============================================================================
# MCP / Genie Polling Functions
# ============================================================================

def poll_genie_response(
    mcp_client,
    tools: list,
    conversation_id: str,
    message_id: str,
    max_attempts: int = 30,
    poll_interval_seconds: int = 2,
    verbose: bool = True
) -> dict:
    """
    Poll for Genie MCP query completion.
    
    Args:
        mcp_client: The DatabricksMCPClient instance
        tools: List of MCP tools (to find poll tool)
        conversation_id: Conversation ID from initial response
        message_id: Message ID from initial response
        max_attempts: Maximum polling attempts
        poll_interval_seconds: Time between polls
        verbose: Whether to print status updates
    
    Returns:
        Dictionary with:
        - success: Whether query completed successfully
        - status: Final status (COMPLETED, FAILED, CANCELLED, TIMEOUT)
        - content: Response content (if successful)
        - raw_result: Raw poll result
    """
    # Find the poll tool
    poll_tool_name = None
    for t in tools:
        if "poll" in t.name.lower():
            poll_tool_name = t.name
            break
    
    if not poll_tool_name:
        return {
            "success": False,
            "status": "ERROR",
            "content": None,
            "raw_result": {"error": "Poll tool not found"}
        }
    
    for attempt in range(max_attempts):
        time.sleep(poll_interval_seconds)
        
        poll_response = mcp_client.call_tool(
            tool_name=poll_tool_name,
            arguments={
                "conversation_id": conversation_id,
                "message_id": message_id
            }
        )
        poll_text = "".join([c.text for c in poll_response.content if hasattr(c, 'text')])
        poll_result = json.loads(poll_text)
        status = poll_result.get("status")
        
        if verbose:
            print(f"   Attempt {attempt + 1}: {status}")
        
        if status == "COMPLETED":
            attachments = poll_result.get("content", {}).get("textAttachments", [])
            return {
                "success": True,
                "status": "COMPLETED",
                "content": attachments,
                "raw_result": poll_result
            }
        elif status in ["FAILED", "CANCELLED"]:
            return {
                "success": False,
                "status": status,
                "content": None,
                "raw_result": poll_result
            }
    
    return {
        "success": False,
        "status": "TIMEOUT",
        "content": None,
        "raw_result": {"error": f"Timeout after {max_attempts} attempts"}
    }


# ============================================================================
# Utility Functions
# ============================================================================

def print_config_summary(config: dict = None) -> None:
    """
    Print a summary of the current configuration.
    
    Args:
        config: Optional pre-loaded config dict
    """
    if config is None:
        config = load_config()
    
    print("=" * 50)
    print("Workshop Configuration Summary")
    print("=" * 50)
    print(f"  Catalog:              {config.get('catalog', 'NOT SET')}")
    print(f"  Schema:               {config.get('schema', 'NOT SET')}")
    print(f"  LLM Endpoint:         {config.get('llm_endpoint_name', 'NOT SET')}")
    print(f"  Embedding Endpoint:   {config.get('embedding_endpoint', 'NOT SET')}")
    print(f"  Embedding Dims:       {config.get('embedding_dims', 'NOT SET')}")
    print(f"  Vector Search:        {config.get('vector_search_endpoint', 'NOT SET')}")
    print(f"  Lakebase Instance:    {config.get('lakebase_instance_name', 'NOT SET')}")
    print(f"  Genie Space ID:       {config.get('genie_space_id', 'NOT SET')}")
    print(f"  Model Name:           {config.get('model_name', 'NOT SET')}")
    print("=" * 50)


def get_full_model_name(config: dict = None) -> str:
    """
    Get the full Unity Catalog model name.
    
    Args:
        config: Optional pre-loaded config dict
    
    Returns:
        Full model name in format catalog.schema.model_name
    """
    if config is None:
        config = load_config()
    
    return f"{config['catalog']}.{config['schema']}.{config['model_name']}"


def get_vector_search_index_name(config: dict = None) -> str:
    """
    Get the full Vector Search index name.
    
    Args:
        config: Optional pre-loaded config dict
    
    Returns:
        Full index name in format catalog.schema.workshop_documents_index
    """
    if config is None:
        config = load_config()
    
    return f"{config['catalog']}.{config['schema']}.workshop_documents_index"


def get_uc_function_names(config: dict = None) -> list:
    """
    Get the list of UC function names for tools.
    
    Args:
        config: Optional pre-loaded config dict
    
    Returns:
        List of fully qualified UC function names
    """
    if config is None:
        config = load_config()
    
    catalog = config['catalog']
    schema = config['schema']
    
    return [
        f"{catalog}.{schema}.simple_calculator",
        f"{catalog}.{schema}.get_weather"
    ]

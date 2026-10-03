"""Loads the tools from the running MCP server through a real MCP client.

The agent never imports the tool functions: it only sees what the server
publishes over HTTP.
"""
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def load_mcp_tools(url: str) -> list[BaseTool]:
    client = MultiServerMCPClient({"eo": {"transport": "streamable_http", "url": url}})
    return await client.get_tools()

What is MCP and how does it standardize tool/resource integration?
Answer:
MCP (Model Context Protocol) is an open standard by Anthropic for seamless integration of external tools and resources with language models, enabling safe, controlled capability expansion.
MCP Architecture:
┌─────────────────┐
│   LLM/Claude    │
├─────────────────┤
│ MCP Client      │  (in Claude.ai, Claude API, IDEs)
│ - Tool Registry │
│ - Resource Mgmt │
└────────┬────────┘
         │ MCP Protocol (JSON-RPC over stdio/HTTP)
┌────────┴────────────────────────────────────────┐
│                                                  │
│  MCP Servers (External Tools & Resources)       │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ File System  │  │  Web Search  │            │
│  │ Server       │  │  Server      │            │
│  └──────────────┘  └──────────────┘            │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │  Database    │  │   APIs       │            │
│  │  Server      │  │  Server      │            │
│  └──────────────┘  └──────────────┘            │
└──────────────────────────────────────────────────┘


Building an MCP Server:

import json
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent, ToolResult

# Create MCP server
server = Server("my-tools-server")

# Define tools
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_database",
        "description": "Search internal database",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
]

# Register tools
@server.list_tools()
async def list_tools():
    return [Tool(**tool_def) for tool_def in TOOLS]

# Implement tool handlers
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> ToolResult:
    """Execute tool and return result"""
    
    if name == "get_weather":
        city = arguments["city"]
        unit = arguments.get("unit", "celsius")
        
        # Call weather API
        weather_data = fetch_weather(city, unit)
        
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(weather_data))],
            is_error=False
        )
    
    elif name == "search_database":
        query = arguments["query"]
        limit = arguments.get("limit", 10)
        
        # Search database
        results = search_db(query, limit)
        
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(results))],
            is_error=False
        )
    
    else:
        return ToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            is_error=True
        )

# Also support resources for read-only data access
@server.list_resources()
async def list_resources():
    """Expose resources (files, data sources)"""
    from mcp.types import Resource
    
    return [
        Resource(
            uri="file:///knowledge-base/docs",
            name="Documentation",
            description="System documentation",
            mimeType="text/markdown"
        ),
        Resource(
            uri="db:///employees",
            name="Employee Database",
            description="Employee records",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content"""
    if uri.startswith("file://"):
        # Return file content
        with open(uri.replace("file://", ""), "r") as f:
            return f.read()
    elif uri.startswith("db://"):
        # Return database resource
        data = query_database(uri.replace("db://", ""))
        return json.dumps(data)

# Run server on stdio
async def main():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server(server):
        # Server runs on stdin/stdout
        await asyncio.Event().wait()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


MCP Client Integration:

# In Claude.ai or via API
import json
from mcp.client.stdio import StdioClientTransport
from mcp.client import Client

# Connect to MCP server
transport = StdioClientTransport(
    command="python",
    args=["/path/to/mcp_server.py"]
)

client = Client(transport)

# List available tools
tools = await client.list_tools()
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"  Description: {tool.description}")
    print(f"  Inputs: {tool.inputSchema}")

# Call tool
result = await client.call_tool(
    name="get_weather",
    arguments={"city": "London", "unit": "celsius"}
)

print(f"Result: {result.content}")



Real-World Example - Database MCP Server:

from mcp.server import Server
from mcp.types import TextContent, ToolResult
import sqlite3
import json

class DatabaseMCPServer:
    def __init__(self, db_path: str):
        self.server = Server("database-tools")
        self.db_path = db_path
        self._register_tools()
    
    def _register_tools(self):
        @self.server.list_tools()
        async def list_tools():
            from mcp.types import Tool
            return [
                Tool(
                    name="query_database",
                    description="Execute SQL SELECT query",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_schema",
                    description="Get database schema",
                    inputSchema={"type": "object", "properties": {}}
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> ToolResult:
            try:
                if name == "query_database":
                    results = self._execute_query(arguments["query"])
                    return ToolResult(
                        content=[TextContent(type="text", text=json.dumps(results))]
                    )
                elif name == "get_schema":
                    schema = self._get_schema()
                    return ToolResult(
                        content=[TextContent(type="text", text=json.dumps(schema))]
                    )
            except Exception as e:
                return ToolResult(
                    content=[TextContent(type="text", text=str(e))],
                    is_error=True
                )
    
    def _execute_query(self, query: str) -> list:
        """Execute query safely"""
        # Validate - only SELECT
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries allowed")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def _get_schema(self) -> dict:
        """Get database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            schema[table] = [
                {"name": col[1], "type": col[2]}
                for col in columns
            ]
        
        conn.close()
        return schema


Usage with Claude API:

# Via Claude API with MCP
from anthropic import Anthropic

client = Anthropic()

# Start conversation
messages = [
    {
        "role": "user",
        "content": "Use the database tool to find all customers"
    }
]

response = client.messages.create(
    model="claude-opus-4-1",
    max_tokens=1024,
    tools=[
        # Tools exposed by MCP servers are registered here
        {
            "name": "query_database",
            "description": "Query the database",
            "input_schema": {...}
        }
    ],
    messages=messages
)

# Handle tool use
if response.stop_reason == "tool_use":
    tool_use = response.content[1]  # Second block is tool use
    tool_name = tool_use.name
    tool_input = tool_use.input
    
    # Execute via MCP
    result = mcp_client.call_tool(tool_name, tool_input)
    
    # Continue conversation
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result
            }
        ]
    })




![8987DD42-E4F8-4D0F-AC2B-7E2EF6088723_1_201_a](https://github.com/user-attachments/assets/b474e658-307b-40c9-9be3-d1873fb84c7d)
![1CEDC74A-EA14-443D-96F1-836C55CC4D1E](https://github.com/user-attachments/assets/0baacc32-13a5-4616-9cb0-44e0fc1a78bf)
![CE2E4F2B-3E29-4472-A537-CB30D6F394C8](https://github.com/user-attachments/assets/583b2576-0f65-4cb6-83a7-6acf26c6ec2f)
![2D9B0305-4907-4DA1-BAA7-78B08C54CC3D](https://github.com/user-attachments/assets/11c0e710-7cca-488d-9bb5-6c9514420922)
![103031A1-207E-43B6-9F7D-6C1C2E0FCB55](https://github.com/user-attachments/assets/a128c480-697b-4e6e-82e1-f88af934134c)





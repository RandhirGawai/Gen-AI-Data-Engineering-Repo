# What is MCP and how does it standardize tool/resource integration?

## Answer

**MCP (Model Context Protocol)** is an open standard by Anthropic for seamless integration of external tools and resources with language models, enabling safe, controlled capability expansion.

---

## MCP Architecture

```text
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
```

---

## Building an MCP Server

```python
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
```

---

## MCP Client Integration

```python
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
```

---

## Real-World Example: Database MCP Server

```python
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
```

---

## Usage with Claude API

```python
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
```
# How to Build an MCP Server (Scenario-Based Question + Step-by-Step Answer)

## Question

> **NovaPay** is a fintech company that processes UPI and card payments. Its support team wants an AI assistant (Claude) to look up transaction status, read the company's refund policy, and help summarize disputes, **without giving the model direct database access or hard-coding custom integrations for each AI client**.
>
> As the engineer, how would you design and build an **MCP (Model Context Protocol) server** for NovaPay? Walk through every step: setup, tools, resources, prompts, running, testing, connecting to Claude, security, and deployment.

---

## Answer Overview

| MCP concept | What it is | NovaPay example |
|---|---|---|
| **Tool** | An action the model can call (can take arguments, may have side effects) | `get_transaction_status(txn_id)` |
| **Resource** | Read-only data the client can load as context | `policy://refunds` |
| **Prompt** | A reusable prompt template the user can trigger | `dispute_summary(txn_id)` |
| **Transport** | How client and server talk (JSON-RPC underneath) | `stdio` (local) or `streamable-http` (remote) |

**Flow:** `Claude (MCP client) <-- JSON-RPC --> NovaPay MCP Server <--> NovaPay systems (DB / APIs)`

---

## Step 1: Define the use case and the surface area

Decide what the model is allowed to do. Keep the first version small and read-only.

- Tool: look up a transaction's status
- Tool: list a customer's recent transactions
- Resource: refund policy document
- Prompt: dispute summary template

**Rule of thumb:** expose the minimum capabilities needed. Never expose raw SQL access.

---

## Step 2: Set up the environment

Requires Python 3.10+.

```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir novapay-mcp && cd novapay-mcp
uv init
uv add "mcp[cli]"
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "mcp[cli]"
```

Project layout:

```text
novapay-mcp/
├── server.py
├── data.py            # mock data / DB access layer
├── pyproject.toml
└── README.md
```

---

## Step 3: Create the data layer

For learning purposes we use mock data. In production, replace this with your database or internal API client.

```python
# data.py
TRANSACTIONS = {
    "TXN1001": {"customer_id": "C01", "amount": 2499.00, "currency": "INR",
                "status": "SUCCESS", "method": "UPI", "created_at": "2026-10-01T10:15:00Z"},
    "TXN1002": {"customer_id": "C01", "amount": 899.00, "currency": "INR",
                "status": "FAILED", "method": "CARD", "created_at": "2026-10-02T14:40:00Z"},
    "TXN1003": {"customer_id": "C02", "amount": 15000.00, "currency": "INR",
                "status": "PENDING", "method": "NETBANKING", "created_at": "2026-10-03T09:05:00Z"},
}

REFUND_POLICY = """# NovaPay Refund Policy
1. Failed transactions are auto-refunded within 5-7 business days.
2. Successful transactions can be refunded on merchant approval.
3. Disputes must be raised within 90 days of the transaction.
"""
```

---

## Step 4: Create the server with `FastMCP`

```python
# server.py
import logging
import sys
from mcp.server.fastmcp import FastMCP

from data import TRANSACTIONS, REFUND_POLICY

# IMPORTANT for stdio: never print() to stdout, it corrupts the protocol.
# Log to stderr instead.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("novapay-mcp")

mcp = FastMCP("novapay-support")
```

---

## Step 5: Add tools

Tools are Python functions. FastMCP builds the JSON schema from the **type hints** and the **docstring**, so write both carefully. The model reads them to decide when to call the tool.

```python
@mcp.tool()
def get_transaction_status(txn_id: str) -> dict:
    """Get the status and details of a NovaPay transaction.

    Args:
        txn_id: Transaction ID, for example TXN1001.
    """
    log.info("get_transaction_status txn_id=%s", txn_id)
    txn = TRANSACTIONS.get(txn_id.upper())
    if not txn:
        raise ValueError(f"Transaction {txn_id} not found")
    return {"txn_id": txn_id.upper(), **txn}


@mcp.tool()
def list_customer_transactions(customer_id: str, limit: int = 10) -> list[dict]:
    """List recent transactions for a customer.

    Args:
        customer_id: Customer ID, for example C01.
        limit: Maximum number of transactions to return (default 10).
    """
    results = [
        {"txn_id": tid, **t}
        for tid, t in TRANSACTIONS.items()
        if t["customer_id"] == customer_id
    ]
    return results[: max(1, min(limit, 50))]   # clamp the limit
```

**Best practices for tools**

- Clear names and docstrings (they act as the model's instructions)
- Strong typing and input validation
- Return structured data (dict or list), not giant text blobs
- Raise errors with helpful messages
- Keep tools idempotent where possible; add confirmation for destructive actions

---

## Step 6: Add resources (read-only context)

Resources are addressed by URI. Use them for documents and reference data.

```python
@mcp.resource("policy://refunds")
def refund_policy() -> str:
    """NovaPay refund policy document."""
    return REFUND_POLICY


# Parameterized resource (URI template)
@mcp.resource("txn://{txn_id}")
def transaction_resource(txn_id: str) -> str:
    """Transaction record as text."""
    txn = TRANSACTIONS.get(txn_id.upper())
    if not txn:
        return f"Transaction {txn_id} not found"
    return "\n".join(f"{k}: {v}" for k, v in txn.items())
```

---

## Step 7: Add prompts (reusable templates)

```python
@mcp.prompt()
def dispute_summary(txn_id: str) -> str:
    """Generate a dispute summary for a transaction."""
    return (
        f"Look up transaction {txn_id} using the get_transaction_status tool, "
        "read the policy://refunds resource, then write a short dispute summary "
        "covering: what happened, whether it is eligible for refund, and next steps."
    )
```

---

## Step 8: Run the server

Add the entry point at the bottom of `server.py`:

```python
if __name__ == "__main__":
    # Local: stdio (Claude Desktop / Claude Code launch it as a subprocess)
    mcp.run(transport="stdio")

    # Remote: uncomment this instead for HTTP
    # mcp.run(transport="streamable-http")
```

Run it:

```bash
uv run server.py
```

For a remote server, configure host and port when creating it:

```python
mcp = FastMCP("novapay-support", host="0.0.0.0", port=8000)
# endpoint will be served at http://<host>:8000/mcp
```

---

## Step 9: Test with the MCP Inspector

```bash
uv run mcp dev server.py
```

This opens a browser UI where you can:

1. See the list of tools, resources, and prompts
2. Call `get_transaction_status` with `TXN1001`
3. Read `policy://refunds`
4. Check error handling with an invalid ID

---

## Step 10: Test programmatically with a client

```python
# test_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="uv", args=["run", "server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "get_transaction_status", {"txn_id": "TXN1001"}
            )
            print("Result:", result.content)

asyncio.run(main())
```

---

## Step 11: Connect to Claude

### Option A: Claude Desktop (local stdio)

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "novapay-support": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/novapay-mcp", "run", "server.py"]
    }
  }
}
```

Restart Claude Desktop. Use **absolute paths**.

### Option B: Claude Code

```bash
# Local stdio server
claude mcp add novapay-support -- uv --directory /absolute/path/to/novapay-mcp run server.py

# Remote HTTP server
claude mcp add --transport http novapay-support https://mcp.novapay.example.com/mcp

claude mcp list
```

### Option C: Claude API (remote servers)

The Claude API can connect to **remote** MCP servers (HTTP) through its MCP connector. Your server must be publicly reachable over HTTPS. Check the current Anthropic docs for the exact request format, since the connector parameters and beta headers can change.

### Try it

Ask Claude: *"What is the status of TXN1002 and is it eligible for a refund?"*
Claude will call `get_transaction_status`, read the policy, and answer.

---

## Step 12: Security and production hardening

| Area | What to do |
|---|---|
| **Authentication** | Use OAuth 2.1 / bearer tokens for remote servers; never leave HTTP servers open |
| **Authorization** | Enforce per-user permissions inside tools; do not trust the model to enforce access |
| **Least privilege** | Read-only DB user; no raw SQL tool; allow-list operations |
| **Input validation** | Validate and sanitize every argument; clamp limits |
| **Data protection** | Mask sensitive fields (card numbers, PII) before returning |
| **Prompt injection** | Treat tool output as untrusted data; do not let returned text trigger actions |
| **Human in the loop** | Require confirmation for write or destructive tools (refunds, reversals) |
| **Audit logging** | Log who called which tool with what arguments (to stderr or a log service) |
| **Rate limiting** | Protect backend systems from runaway tool loops |
| **Secrets** | Environment variables or a secret manager, never in code |

---

## Step 13: Deploy

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir "mcp[cli]"
COPY . .
EXPOSE 8000
CMD ["python", "server.py"]
```

(Set `mcp.run(transport="streamable-http")` in `server.py` for this container.)

```bash
docker build -t novapay-mcp .
docker run -p 8000:8000 novapay-mcp
```

Then:

- Put it behind HTTPS (reverse proxy, API gateway, or load balancer)
- Deploy to Azure Container Apps / AWS ECS / Kubernetes
- Add health checks, monitoring, and alerting
- Version your tools carefully; changing a tool's schema can break clients

---

## Quick Recap Checklist

1. Define scope (tools, resources, prompts)
2. Set up Python env and install `mcp[cli]`
3. Build the data/API layer
4. Create `FastMCP` server
5. Add tools with typed args and docstrings
6. Add resources
7. Add prompts
8. Choose transport and run
9. Test in MCP Inspector
10. Test with a programmatic client
11. Connect to Claude Desktop, Claude Code, or the API
12. Secure it (auth, least privilege, validation, audit)
13. Containerize and deploy over HTTPS

---

## Common Pitfalls

- Using `print()` in a stdio server, which breaks the protocol (log to stderr)
- Relative paths in client config (use absolute paths)
- Vague tool descriptions, so the model never picks the right tool
- Returning huge payloads that waste context
- Exposing write operations without confirmation or authorization

![8987DD42-E4F8-4D0F-AC2B-7E2EF6088723_1_201_a](https://github.com/user-attachments/assets/b474e658-307b-40c9-9be3-d1873fb84c7d)
![1CEDC74A-EA14-443D-96F1-836C55CC4D1E](https://github.com/user-attachments/assets/0baacc32-13a5-4616-9cb0-44e0fc1a78bf)
![CE2E4F2B-3E29-4472-A537-CB30D6F394C8](https://github.com/user-attachments/assets/583b2576-0f65-4cb6-83a7-6acf26c6ec2f)
![2D9B0305-4907-4DA1-BAA7-78B08C54CC3D](https://github.com/user-attachments/assets/11c0e710-7cca-488d-9bb5-6c9514420922)
![103031A1-207E-43B6-9F7D-6C1C2E0FCB55](https://github.com/user-attachments/assets/a128c480-697b-4e6e-82e1-f88af934134c)





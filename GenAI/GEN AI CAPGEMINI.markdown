# AI and ECS FAQ


Temperature = how “wild” or “serious” they are.

Top-k / Top-p = how many words they keep in their vocabulary while choosing the next word.

Max tokens = how long they’re allowed to talk.

Repetition penalty = telling them “don’t repeat yourself too much.”


Inside your Docker container:

📦 Ubuntu (lightweight OS)
 ├── Python 3.10
 ├── Flask app (chatbot.py)
 ├── Model weights (optional, or API integration code)
 ├── Libraries (transformers, langchain, faiss, etc.)
 ├── Configs (API keys, database URI)
 └── Gunicorn/Uvicorn (to serve the chatbot API)

## 1. How to Perform Indexing of Context (Technique Behind It)?

Indexing of context usually refers to preparing documents for retrieval in RAG (Retrieval-Augmented Generation).

**Techniques:**
- **Text Splitting / Chunking**: Break large documents into smaller chunks (e.g., 500–1000 tokens) for processing.
- **Embeddings**: Convert chunks into vector representations using embedding models (e.g., OpenAI text-embedding, sentence-transformers).
- **Vector Indexing**: Store vectors in a vector database (e.g., FAISS, Pinecone, Weaviate, OpenSearch KNN) for fast similarity search.
- **Search Algorithms**: Use cosine similarity, dot product, or ANN (Approximate Nearest Neighbor) algorithms like HNSW.

**Summary**: Indexing = chunking + embeddings + vector database indexing.

### What is HNSW (Hierarchical Navigable Small World)?

HNSW is an efficient method to find similar items in large datasets, commonly used in vector databases for fast document/embedding retrieval.

**How it Works (Simplified):**
- **Layers like Maps**:
  - Data is organized in layers.
  - Top layer: Few points (rough overview).
  - Middle layers: More details.
  - Bottom layer: All points (full detail).
  - Search starts at the top and moves down, like zooming in on Google Maps.
- **Connections (Shortcuts)**:
  - Each point connects to nearby points and has shortcuts to far-away points.
  - Shortcuts enable quick jumps across the dataset.
- **Search Process**:
  - Start at a random point in the top layer.
  - Move to the closest neighbor, descending layer by layer.
  - At the bottom, search only nearby points for the best matches.
- **Efficiency**:
  - Logarithmic time complexity (O(log N)) vs. linear search (O(N)).
- **Analogy**:
  - Finding a café in a city:
    - Top layer = highways (get to the right area).
    - Middle layer = district roads (find the neighborhood).
    - Bottom layer = street map (walk to the café).
    - Shortcuts = expressways for quick jumps.
- **Why Popular**:
  - Fast and accurate for millions of points.
  - Used in FAISS, Pinecone, Milvus, Weaviate, OpenSearch, etc.

## 2. If Model Starts Hallucinating, What Can You Do?

Hallucination occurs when a model generates incorrect or fabricated information.

**Ways to Reduce**:
- **Ground with External Knowledge**: Use RAG (vector DB + context injection).
- **Prompt Engineering**: Instruct explicitly (e.g., “Answer strictly from provided context. If not found, say ‘Not available’”).
- **Fine-tuning / LoRA**: Train with domain-specific data to reduce knowledge gaps.
- **Post-processing**: Validate outputs with rules, regex, or factual APIs.
- **Confidence Scores**: Use embedding similarity scores to reject low-confidence answers.

## 3. If Model Starts Answering Randomly, What Can You Do?

Random answers indicate a lack of grounding or context.

**Solutions**:
- Ensure the retrieval pipeline works correctly (documents are retrieved).
- **Temperature Control**: Set low temperature (0.0–0.3) for deterministic answers.
- **Guardrails**: Use tools like Guardrails AI, LangChain constraints, or function-calling.
- **Log Queries**: Debug to identify missing context or poor prompts.

## 4. What is the Prerequisite for ECS (Elastic Container Service)?

Before running workloads on AWS ECS:

- **Dockerized Application**: A Docker image is required.
- **ECS Cluster**: Create with EC2 or Fargate.
- **IAM Role & Permissions**: ECS task execution role.
- **Task Definition**: JSON file defining containers, CPU, memory, networking.
- **Networking Setup**: VPC, Subnets, Security groups.
- **(Optional)**: Load Balancer & Auto-scaling policies.

**Summary**: Prerequisites = Docker + ECS Cluster + IAM + Networking + Task Definition.

## 5. What are Components of MCP?

Assuming MCP refers to **Model Context Protocol** (Anthropic standard, given AI context):

**Components**:
- **Server**: Exposes tools/resources (APIs, DB, functions).
- **Client**: LLM or application using MCP to call the server.
- **Transport**: Standard channel (WebSockets, JSON-RPC).
- **Stdio / Non-stdio Modes**: Defines client-server communication method.

## 6. If There is a Sudden Spike of Users in Chatbot, What Can You Do?

To handle scalability and fault tolerance:

- **Auto Scaling**: Use ECS/EKS/Kubernetes HPA or AWS Lambda scaling.
- **Load Balancer (ALB/NLB)**: Distribute load.
- **Caching**: Store repeated query results (Redis, CloudFront).
- **Queueing (SQS/Kafka)**: Handle burst traffic with async processing.
- **Rate Limiting / Throttling**: Prevent overload.
- **Monitoring**: Use CloudWatch/Prometheus for alerts.

## 7. What is STDIO and Non-STDIO in MCP?

In **Model Context Protocol (MCP)**:

- **STDIO (Standard Input/Output Mode)**:
  - Client and server communicate via process pipes (stdin/stdout).
  - Example: Local tools on a developer’s machine.
- **Non-STDIO (Remote Mode)**:
  - Client communicates via sockets, APIs, or other transports.
  - Example: Cloud-hosted tools.

**Summary**: STDIO = local, process-level. Non-STDIO = remote, network-level.

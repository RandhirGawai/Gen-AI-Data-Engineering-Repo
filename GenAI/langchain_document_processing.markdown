# Comprehensive LangChain Document Processing and RAG Pipeline Guide

This guide provides a complete overview of LangChain components for document loading, text splitting, embedding generation, vector storage, retrieval, and advanced techniques like LCEL, LangServe, LangGraph, and agentic RAG workflows. It includes all provided examples, explanations, and concepts for building a unified ingestion pipeline.

## 1. Document Loaders

Document loaders fetch and parse data from various sources into LangChain's document format.

### 1.1 Text Loader
Loads plain `.txt` files from local storage.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/sample.txt")
documents = loader.load()
print(documents[0].page_content)  # View file content
```

### 1.2 Web-Based Loader
Scrapes and loads content from web pages.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
documents = loader.load()
for doc in documents:
    print(doc.metadata, doc.page_content[:200])
```

### 1.3 PDF Loader
Extracts text from PDF files, treating each page as a separate document.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()
print(len(documents))  # Number of pages as separate docs
```

### 1.4 S3 Directory Loader
Loads files from an AWS S3 bucket.

```python
from langchain_community.document_loaders import S3DirectoryLoader

loader = S3DirectoryLoader(
    bucket="my-bucket-name",
    prefix="data/documents/",  # folder path inside S3
    aws_access_key_id="YOUR_KEY",
    aws_secret_access_key="YOUR_SECRET"
)
documents = loader.load()
```

### 1.5 Azure Blob Storage Loader
Loads files from an Azure Blob Storage container.

```python
from langchain_community.document_loaders import AzureBlobStorageFileLoader

loader = AzureBlobStorageFileLoader(
    conn_str="DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net",
    container="my-container",
    blob_name="sample.pdf"
)
documents = loader.load()
```

**Tips for All Loaders**:
- After loading, split text into chunks using `RecursiveCharacterTextSplitter`.
- Normalize metadata (e.g., source, page number) for consistency.
- Use async loaders (`.aload()`) for large-scale ingestion to improve performance.

**Unified Pipeline Option**: A single pipeline can load from TXT, Web, PDF, S3, or Azure Blob, automatically select the right splitter, and store into a vector database for retrieval-augmented generation (RAG).

## 2. Text Splitters

Text splitters break large documents into smaller chunks for embedding and retrieval.

### 2.1 RecursiveCharacterTextSplitter
Splits text without breaking words or sentences, using multiple separators in priority order (paragraph → sentence → word → character).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Max characters per chunk
    chunk_overlap=50,    # Overlap between chunks for context
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_text("Your large text here...")
print(docs)
```

**When to use**:
- ✅ General text (PDF, TXT, scraped content).
- ✅ Preserves semantic meaning by avoiding mid-word breaks.
- Best for most document types.

### 2.2 CharacterTextSplitter
Splits text purely by character count, ignoring sentence or paragraph boundaries.

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"  # split at newline if possible
)
docs = text_splitter.split_text("Your large text here...")
print(docs)
```

**When to use**:
- ✅ Structured or semi-structured text where splitting location doesn’t matter.
- ❌ Not ideal for conversational or semantic search due to potential mid-sentence breaks.

### 2.3 HTMLHeaderTextSplitter
Parses HTML and splits based on header tags (`<h1>`, `<h2>`, etc.), keeping sections under their respective headers.

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_text = """
<h1>Introduction</h1>
<p>This is intro text.</p>
<h2>Background</h2>
<p>Details about background.</p>
"""
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2")
]
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(html_text)
print(docs)
```

**When to use**:
- ✅ Web pages, blog posts, technical documentation with HTML headings.
- ✅ Keeps context grouped by topic.

### 2.4 RecursiveJsonSplitter
Splits nested JSON into smaller pieces while preserving structure.

```python
from langchain_text_splitters import RecursiveJsonSplitter

json_data = {
    "user": {
        "name": "Alice",
        "bio": "Data scientist with experience in ML.",
        "projects": [
            {"name": "Project A", "desc": "AI model"},
            {"name": "Project B", "desc": "Data pipeline"}
        ]
    }
}
splitter = RecursiveJsonSplitter(max_chunk_size=50)
docs = splitter.split_json(json_data)
print(docs)
```

**When to use**:
- ✅ Large JSON logs, configs, or API responses.
- ✅ Keeps related key-value pairs together.

**Summary Table**:

| Splitter Type               | Best For                     | Preserves Meaning? | Structure Aware? |
|-----------------------------|------------------------------|--------------------|------------------|
| RecursiveCharacterTextSplitter | General text                | ✅                 | ❌               |
| CharacterTextSplitter        | Raw text chunks             | ❌                 | ❌               |
| HTMLHeaderTextSplitter       | HTML docs                   | ✅                 | ✅               |
| RecursiveJsonSplitter        | JSON data                   | ✅                 | ✅               |

**Unified Pipeline Note**: A single ingestion pipeline can automatically choose the right splitter based on file type (e.g., `HTMLHeaderTextSplitter` for HTML, `RecursiveJsonSplitter` for JSON).

## 3. Embeddings

Embeddings convert text into numerical vectors for similarity search and RAG.

### 3.1 OpenAI Embeddings
Hosted API, optimized for semantic search, clustering, and RAG. Common models: `text-embedding-3-small`, `text-embedding-3-large`.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))  # embedding length
```

**Pros**:
- ✅ High accuracy.
- ✅ Well-maintained.
**Cons**:
- ❌ Requires API key and internet.
- ❌ Paid after free credits.

### 3.2 Hugging Face Embeddings
Open-source, free, with many models (e.g., `sentence-transformers/all-MiniLM-L6-v2`).

**Local Example**:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))
```

**Hugging Face Inference API Example**:

```python
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key="YOUR_HF_API_KEY",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))
```

**Pros**:
- ✅ Free, runs locally.
- ✅ Many pre-trained models.
**Cons**:
- ❌ Larger models can be slow without GPU.

### 3.3 LLaMA-based Embeddings
Uses fine-tuned LLaMA-based models (e.g., BGE, Instructor) via Ollama.

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama2")
vector = embeddings.embed_query("This is an example sentence")
print(len(vector))
```

**Example Models**:
- `BAAI/bge-large-en-v1.5`: High quality.
- `hkunlp/instructor-xl`: Task-aware embeddings.

**Pros**:
- ✅ Runs fully offline.
- ✅ Customizable with fine-tuning.
**Cons**:
- ❌ Needs strong hardware for big models.

**Comparison Table**:

| Provider        | Example Model                  | Dim  | Offline? | Cost | Quality |
|-----------------|-------------------------------|------|----------|------|---------|
| OpenAI          | text-embedding-3-large        | 3072 | ❌       | Paid | ⭐⭐⭐⭐⭐ |
| Hugging Face    | all-MiniLM-L6-v2             | 384  | ✅       | Free | ⭐⭐⭐⭐  |
| LLaMA-based     | BGE-large                    | 1024 | ✅       | Free | ⭐⭐⭐⭐  |

## 4. Vector Databases and Retrieval

Vector databases store embeddings and enable fast similarity searches.

### 4.1 Vector Databases

- **Chroma**: Open-source, local-first, ideal for prototyping RAG pipelines.
  ```python
  from langchain_chroma import Chroma
  db = Chroma(collection_name="docs")
  ```

- **FAISS**: Fast, in-memory or disk-based, scales to millions of vectors.
  ```python
  from langchain_community.vectorstores import FAISS
  db = FAISS.from_texts(["doc1", "doc2"], embedding=embeddings)
  ```

- **Hugging Face Hub Datasets**: Stores precomputed embeddings, good for sharing public datasets (not live search).

### 4.2 Retrieval Techniques

- **Similarity Search**: Finds vectors closest to the query vector (cosine, dot product, L2).
  ```python
  results = db.similarity_search("query", k=5)
  ```

- **Max Marginal Relevance (MMR)**: Balances relevance and diversity to avoid redundant results.
  ```python
  results = db.max_marginal_relevance_search("query", k=5)
  ```

- **Filter-based Search**: Filters results by metadata (e.g., document type, date).
  ```python
  results = db.similarity_search("query", filter={"type": "pdf"})
  ```

**Why Retrieval?**  
- Pre-processes queries (rephrasing, embedding creation).  
- Post-processes results (re-ranking, filtering, metadata enrichment).  
- Integrates with LLM chains for contextual answering.

**L2 Score in Similarity Search**  
- L2 = Euclidean distance: `√∑(xi - yi)²`.  
- Lower L2 = more similar.  
- Used in FAISS for speed and simplicity in Euclidean space, especially when vectors are not normalized.

**all_dangerous_deserialization**  
- Allows loading arbitrary pickled Python objects (e.g., FAISS index configs).  
- **Warning**: Dangerous due to potential execution of arbitrary code. Only enable for trusted sources. Prefer JSON for metadata.

## 5. LangChain Components

### 5.1 LangChain OpenAI
Integration layer for OpenAI APIs (chat models, embeddings). Handles API authentication, prompt formatting, token counting, and retry logic.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
response = llm.invoke("Write a haiku about the ocean")
print(response.content)
```

### 5.2 LangChain Tracking
Tracks prompts, inputs, outputs, errors, and timing via LangSmith for debugging and monitoring.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key"
```

### 5.3 LangChain Project
Groups related runs, datasets, and experiments in LangSmith for a specific application.

```python
os.environ["LANGCHAIN_PROJECT"] = "financial-chatbot"
```

### 5.4 ChatPromptTemplate
Creates structured, reusable prompts with placeholders for dynamic values.

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question:\n{question}"
)
message = prompt.format_messages(question="What is the capital of France?")
print(message)
```

### 5.5 Output Parser
Converts raw LLM output into structured formats (e.g., JSON, lists, Python objects).

```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

schemas = [
    ResponseSchema(name="answer", description="The answer to the question"),
    ResponseSchema(name="source", description="Where the answer comes from")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
prompt_text = parser.get_format_instructions()
print(prompt_text)
```

**Summary Table**:

| Term                  | Purpose                                              |
|-----------------------|-----------------------------------------------------|
| LangChain OpenAI      | Integrates OpenAI models into LangChain pipelines    |
| LangChain Tracking    | Logs & monitors chain executions (via LangSmith)    |
| LangChain Project     | Groups related runs/experiments into one workspace   |
| ChatPromptTemplate    | Creates dynamic, reusable prompts for chat models    |
| Output Parser         | Converts messy LLM output into structured data       |

## 6. Document Processing Chains

### 6.1 create_stuff_document_chain
Combines multiple retrieved documents into a single prompt for the LLM.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Step 1: Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Step 2: Define Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. 
Answer the question based on the following documents:
{context}

Question: {question}
""")

# Step 3: Create Stuff Chain
chain = create_stuff_documents_chain(llm, prompt)

# Step 4: Run the chain
docs = [
    {"page_content": "Paris is the capital of France."},
    {"page_content": "It is known for the Eiffel Tower."}
]
response = chain.invoke({"question": "What is the capital of France?", "context": docs})
print(response)
```

**Why “Stuff” Approach?**  
- **Pros**: Simple, direct—combines all context into one prompt.  
- **Cons**: Can fail with large documents due to token limits; no ranking or summarization.  

**Other Strategies**:
- **Map-Reduce**: Summarize each document, then summarize summaries.
- **Refine**: Generate an answer from the first document, iteratively refine with others.

## 7. LangChain Expression Language (LCEL)

LCEL is a declarative way to compose LangChain components (LLMs, prompts, retrievers, parsers) into pipelines.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
parser = StrOutputParser()
chain = prompt | llm | parser
result = chain.invoke({"text": "Hello World"})
print(result)  # Bonjour le monde
```

**Why LCEL?**  
- Clean, functional-style syntax.  
- Easier debugging and testing.  
- Supports `.invoke()`, `.batch()`, `.stream()` for flexible execution.

**LangChain Core**  
- Minimal, dependency-light foundation of LangChain.  
- Includes base classes for LLMs, prompts, document loaders, output parsers, and tracing.  
- **Why use it?**: Lightweight, faster imports, easier to build custom components.

**Why OutputParser?**  
- Converts messy LLM output into structured formats (e.g., JSON, Pydantic models).  
- Prevents formatting hallucinations and ensures machine-readable outputs.

```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

schemas = [
    ResponseSchema(name="answer", description="The main answer"),
    ResponseSchema(name="source", description="Where the answer came from")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()
prompt_text = f"""
Answer the question and respond in the format below:
{format_instructions}

Question: What is the capital of France?
"""
```

## 8. LangServe

LangServe deploys LangChain pipelines as APIs using FastAPI.

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

app = FastAPI()
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm
add_routes(app, chain, path="/joke")
```

**Usage**: POST to `/joke` with `{"topic": "cats"}`.  
**Benefits**:  
- Auto-generated API docs (Swagger/OpenAPI).  
- Supports streaming, batching, and authentication.  
- Makes chains callable by external apps (web, mobile, CLI).

**Chains as API**: Exposes LangChain chains as endpoints for integration with frontends (React, Streamlit) or other systems.

**trim_messages**  
- Shortens conversation history to fit LLM token limits.  
- Can remove oldest messages, truncate long messages, or preserve system prompts.

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import trim_messages

messages = [
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!"),
    HumanMessage(content="Tell me a long story about AI...")
]
trimmed = trim_messages(messages, max_tokens=100, strategy="last")
```

## 9. LangGraph Concepts

# Key GenAI Concepts Simplified for Interviews

This guide simplifies key GenAI concepts in a logical sequence, starting with foundational state management and progressing to advanced techniques. It includes the use of `TypedDict` and other critical topics, tailored for clear communication in technical interviews.

## 1. TypedDict
**What it is**: A Python type hint for defining dictionaries with specific keys and value types, ensuring type safety for state management.

**Example**:
```python
from typing_extensions import TypedDict

class State(TypedDict):
    query: str
    messages: list[str]
```

**How it works**:
- Defines a schema for state with fixed keys and types.
- Provides IDE support and type checking without the boilerplate of dataclasses.

**Use Case**: Managing structured state in workflows (e.g., chatbot state with query and messages).

**Interview Tips**:
- Highlight `TypedDict` for lightweight, type-safe state definitions.
- Compare with dataclasses (more features, more code) and Pydantic (validation, more overhead).
- **Pitfall**: Lacks runtime validation; use Pydantic if validation is critical.

## 2. Reducer
**What it is**: Logic that merges state updates from multiple nodes into a consistent state, like a "combine" step in MapReduce.

**How it works**:
- Each state key (e.g., `messages`, `bar`) can have a custom reducer.
- When multiple nodes update the same key, the reducer merges them.
- Example:
```python
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], operator.add]  # Reducer for appending lists

# Node1: {"bar": ["A"]}
# Node2: {"bar": ["B"]}
# Result after reducer: {"bar": ["A", "B"]}
```

**Use Case**: Aggregating chat messages, logs, or search results.

**Interview Tips**:
- Explain how reducers prevent data loss (e.g., appending vs. overwriting).
- **Pitfall**: Incorrect reducers can overwrite data; stress testing merge logic.

## 3. Annotation Function
**What it is**: Attaches a reducer to a state field using Python’s `Annotated` type, specifying how updates are merged.

**Example**:
```python
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage

messages: Annotated[list[AnyMessage], add_messages]
```
- The `add_messages` reducer appends messages, avoids duplicates, and handles IDs.

**Use Case**: Managing growing chat histories in conversational AI.

**Interview Tips**:
- Emphasize annotations for type-safe, custom merge logic.
- **Pitfall**: Without annotations, updates may overwrite earlier data.

## 4. Dataclasses
**What it is**: An alternative to `TypedDict` for state schemas, offering stronger type safety and IDE support.

**Example**:
```python
from dataclasses import dataclass

@dataclass
class State:
    query: str
    messages: list[str]
```

**Use Case**: Large projects needing strict type checking.

**Interview Tips**:
- Compare dataclasses (more type safety) vs. `TypedDict` (less code).
- **Pitfall**: More boilerplate; justify for complex projects.

## 5. ChatMsg (Messages)
**What it is**: Represents conversation history (human or AI messages), managed by the `add_messages` reducer.

**Example**:
```python
from langchain_core.messages import HumanMessage, AIMessage

history = [
    HumanMessage(content="What's the weather?"),
    AIMessage(content="It’s sunny today.")
]
```

**Use Case**: Tracks multi-step conversation workflows.

**Interview Tips**:
- Explain its role in maintaining chatbot context.
- **Pitfall**: History can grow large; suggest pruning or summarizing.

## 6. Stream
**What it is**: Enables real-time, token-level output for partial results.

**Example**:
```python
async for chunk in app.stream(input_data):
    print(chunk)  # Prints partial tokens
    app.update({"messages": chunk["messages"]})
```

**Benefits**:
- Real-time user feedback.
- Progress tracking for long tasks.
- Human-in-the-loop adjustments.

**Interview Tips**:
- Highlight streaming’s role in user experience.
- **Pitfall**: Can overload logs; suggest buffering or throttling.

## 7. LLM in Graph Nodes
**What it is**: A graph node that calls an LLM to process state and output new state, acting as an "agent brain."

**Example**:
```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

def ask_model(state: State):
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke({"inputs": state["messages"]})
    return {"messages": [AIMessage(content=response.content)]}
```

**Use Case**: Multi-step reasoning in pipelines.

**Interview Tips**:
- Show how LLMs integrate with workflows.
- **Pitfall**: Without a reducer, messages may overwrite.

## 8. Tool Binding
**What it is**: Binds external functions (tools) to an LLM for extended capabilities.

**Example**:
```python
from langchain.tools import tool

@tool
def get_weather(city: str):
    return f"The weather in {city} is sunny."

agent = agent.bind_tools([get_weather])
```

**Use Case**: APIs, database queries, file access.

**Interview Tips**:
- Describe tools as "superpowers" for LLMs.
- **Pitfall**: LLMs may hallucinate tool calls; stress input validation.

## 9. Tool Calls from LLM
**What it is**: LLM decides to call a tool, executes it, and updates state (ReAct: Reason → Act → Observe → Repeat).

**Example Flow**:
1. User: “What’s the weather in Delhi?”
2. LLM: Intent → `get_weather(city="Delhi")`
3. Node executes → State updates with “The weather in Delhi is sunny.”
4. LLM generates final answer.

**Use Case**: Dynamic tool usage in agents.

**Interview Tips**:
- Explain the ReAct loop clearly.
- **Pitfall**: Invalid tool calls need a controller.

## 10. Router
**What it is**: A node that directs workflow to different nodes/subgraphs based on conditions.

**Example**:
```python
from langgraph.graph import StateGraph

graph = StateGraph()

def route(state):
    if "math" in state["query"]:
        return "math_node"
    else:
        return "chat_node"

graph.add_router("router_node", route)
```

**Use Case**: Directing math queries to a calculator, chat to LLM.

**Interview Tips**:
- Stress clear routing for scalability.
- **Pitfall**: Weak conditions can misroute queries.

## 11. Conditional Edges
**What it is**: Routes from one node to another based on conditions.

**Example**:
```python
graph.add_conditional_edges(
    "router_node",
    condition=lambda state: state["intent"] == "math",
    target="math_node"
)
```

**Use Case**: Multi-branch workflows.

**Interview Tips**:
- Highlight flexibility in dynamic workflows.
- **Pitfall**: Overuse leads to complex graphs.

## 12. Agent Architecture (ReAct)
**What it is**: ReAct combines reasoning (planning, task decomposition) and acting (tool calls). Iterates: Reason → Act → Observe → Repeat.

**How it works**:
- Reason: Plan next step (e.g., “Need author → search”).
- Act: Call tool (e.g., search, calculator).
- Observe: Review output.
- Repeat: Update plan or answer.

**Use Case**: Reduces hallucinations by grounding in tool outputs.

**Interview Tips**:
- Explain how ReAct improves reliability.
- Mention hiding reasoning in production for security.

## 13. Agent Memory
**What it is**: Stores context or intermediate results.

**Types**:
- **ConversationBufferMemory**: Raw chat history.
- **ConversationSummaryMemory**: Summarized history to save tokens.
- **VectorStoreRetrieverMemory**: Retrieves facts via embeddings.

**Example**: Remembers “flight to New York” for “Make it business class.”

**Interview Tips**:
- Discuss memory’s role in context continuity.
- **Pitfall**: Large histories increase tokens; suggest summarization.

## 14. astream
**What it is**: Asynchronous streaming for non-blocking, real-time outputs.

**Example**:
```python
async for event in agent.astream({"query": "What's the weather?"}):
    print(event)
```

**Use Case**: Real-time responses in async workflows.

**Interview Tips**:
- Highlight non-blocking benefits for scalability.
- **Pitfall**: Requires careful handling to avoid flooding.

## 15. Advanced Workflow Techniques
### Prompt Chaining
- Breaks tasks into steps: Generate → Improve → Polish.
- **Use Case**: Writing blogs, refining answers.

### Parallelization
- Runs nodes concurrently for speed.
- **Use Case**: Embedding documents, API calls in parallel.

### Orchestrator Workflow
- Manages conversation flow, node execution, and results.

### Send API
- Passes messages/data between nodes asynchronously.
- **Example**:
```python
await graph.send("node_name", input_data)
```

### Evaluator/Optimizer
- Evaluator: Measures performance (e.g., accuracy).
- Optimizer: Adjusts prompts or retrieval based on feedback.

**Interview Tips**:
- Emphasize how these techniques improve efficiency and quality.

## 16. Advanced RAG Techniques
### Agentic RAG
- Combines RAG with agent reasoning for dynamic retrieval.
- **Use Case**: Multi-step Q&A in customer support.

### Corrective RAG
- Self-reflects, grades answers, and re-retrieves if needed.
- **Use Case**: Reduces hallucinations in factual queries.

### Adaptive RAG
- Adjusts retrieval based on query complexity (e.g., adaptive k-value, reranking).
- **Use Case**: Balances speed and accuracy.

**Interview Tips**:
- Explain how RAG enhances LLM reliability.

## 17. Unified Ingestion Pipeline
**What it is**: A scalable pipeline for processing data, generating embeddings, and enabling advanced RAG.

**Components**:
- Load data (TXT, PDF, HTML, S3, Azure Blob).
- Use splitters (e.g., `RecursiveCharacterTextSplitter`).
- Generate embeddings (OpenAI, Hugging Face).
- Store in vector databases (Chroma, FAISS).
- Support similarity search, MMR, or filter-based retrieval.
- Integrate with LangChain (LCEL, LangServe, LangGraph).

**Interview Tips**:
- Highlight scalability and production-readiness.
- Discuss flexibility for diverse data sources.

## Interview Preparation Tips
- **Logical Sequence**: Start with state management (`TypedDict`, reducers), then workflows (routers, conditional edges), and finally advanced techniques (RAG, pipelines).
- **Use Analogies**: E.g., reducer as a “merge manager,” ReAct as “think-act-learn.”
- **Show Practicality**: Tie concepts to real-world use cases (chatbots, search).
- **Address Pitfalls**: Demonstrate awareness of errors and solutions.
- **Highlight Trade-offs**: Compare `TypedDict` vs. dataclasses, streaming vs. batch.
- **Be Concise**: Practice explaining each concept in 1–2 minutes.

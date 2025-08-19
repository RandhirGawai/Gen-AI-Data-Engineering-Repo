# Detailed Retrieval-Augmented Generation (RAG) Pipeline and Tokenization

This document provides an in-depth explanation of the **Retrieval-Augmented Generation (RAG)** pipeline, covering **text splitting**, **embedding models**, **vector stores**, **retrievers**, and **large language models (LLMs)**. It emphasizes **tokenization**, including **token limits** for specific models, **overlap**, **spilling**, and the role of **multidimensional vectors** in NLP. The notes are designed to be comprehensive, beginner-friendly, and include practical examples, a cheat sheet, and additional insights for better understanding.

---

## 1. Text Splitting (Preprocessing)

### Goal
Break large text content into smaller, meaningful chunks to fit within the token limits of LLMs and improve retrieval accuracy for tasks like question answering or summarization.

### Why It’s Important
- **Token Limits**: LLMs have fixed input sizes, which vary by model (see Section 1.4 for specific token limits). Splitting ensures long texts are processed within these constraints.
- **Improved Retrieval**: Smaller chunks allow precise matching of relevant content during similarity search, reducing noise.
- **Common Problems Solved**:
  - **Input Too Long**: Long documents exceed model token limits, requiring splitting or truncation.
  - **Unstructured Splits**: Splitting mid-sentence can break context, leading to poor model performance. Overlap helps mitigate this.
  - **Spilling**: Excess tokens that don’t fit in a chunk “spill” into the next chunk or are truncated, requiring careful handling.

### Steps to Build
1. **Choose a Text Splitter**:
   - **RecursiveCharacterTextSplitter**: Default choice, splits on characters (e.g., newlines, spaces) while preserving sentence boundaries where possible. Best for general text like articles or books.
   - **MarkdownHeaderTextSplitter**: Splits Markdown or README files based on headers (e.g., `#`, `##`), ideal for structured documents.
   - **LanguageTextSplitter**: Designed for source code, respects syntax of languages like Python, JavaScript, or Java.
   - **HTMLHeaderTextSplitter**: Splits scraped HTML pages based on tags (e.g., `<h1>`, `<p>`), suitable for web content.
   - **TokenTextSplitter**: Splits based on token counts, useful for precise control over token limits.
   - **NLTKTextSplitter/SpacyTextSplitter**: NLP-aware splitters that use linguistic rules (e.g., sentence boundaries).
2. **Configure Parameters**:
   - **chunk_size**: Maximum length of each chunk, measured in tokens or characters (e.g., 500 tokens or 1000 characters).
   - **chunk_overlap**: Number of overlapping tokens between consecutive chunks to preserve context (e.g., 50 tokens, or 10–20% of `chunk_size`).
   - **separators**: For `RecursiveCharacterTextSplitter`, specify separators (e.g., `["\n\n", "\n", " "]`) to prioritize splitting at natural breaks.
3. **Split the Text**:
   - Use the `.split_text(text)` method to generate chunks, ensuring overlap and handling spilling.

### Token Limits for Popular Models
- **BERT-based Models** (e.g., `bert-base-uncased`): 512 tokens.
- **RoBERTa**: 512 tokens.
- **DistilBERT**: 512 tokens.
- **T5 (Text-to-Text Transfer Transformer)**: 512 tokens (input), adjustable for output.
- **GPT-3 (text-davinci-003)**: 4096 tokens (combined input and output).
- **GPT-4 (gpt-4)**: 8192 tokens, with some variants supporting 32k or 128k.
- **LLaMA (LLaMA-2-7B)**: 2048 tokens (varies by implementation).
- **Mistral (Mistral-7B)**: 8192 tokens.
- **Grok 3 (xAI)**: Varies, typically 8192 tokens for standard inputs (exact limit depends on configuration).

### Handling Overlap and Spilling
- **Overlap**: Ensures context continuity by including shared tokens between chunks. For example:
  - Document: 1000 tokens, model limit: 512 tokens, overlap: 50 tokens.
  - Chunks:
    - Chunk 1: Tokens 1–512.
    - Chunk 2: Tokens 463–974 (overlapping tokens 463–512).
    - Chunk 3: Tokens 925–1000 (padded with 487 `[PAD]` tokens if needed).
  - **Stride**: The step size between chunks (e.g., 512 - 50 = 462 tokens).
- **Spilling**: Tokens exceeding the model’s limit (e.g., tokens 513–1000 in a 512-token model) are either:
  - **Truncated**: Discarded, risking loss of critical information.
  - **Moved to Next Chunk**: Processed in subsequent chunks, often with overlap to retain context.
- **Padding**: Short chunks are padded with `[PAD]` tokens to match the model’s input size (e.g., a 200-token chunk padded with 312 `[PAD]` tokens for BERT).

### Code Example
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "Your very long article here..."  # Assume 1000 tokens
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Max 500 tokens per chunk
    chunk_overlap=50,  # 50 tokens overlap
    separators=["\n\n", "\n", " ", ""]  # Prioritize splitting at paragraphs, lines, spaces
)
chunks = splitter.split_text(text)
print(f"Number of chunks: {len(chunks)}")
print(chunks[0][:100])  # First 100 characters of first chunk
```

### Additional Insights
- **Choosing chunk_size**: Align with the model’s token limit (e.g., 500 for BERT’s 512, leaving room for special tokens like `[CLS]`, `[SEP]`).
- **Overlap Size**: Typically 10–20% of `chunk_size` (e.g., 50–100 tokens for a 500-token chunk) to balance context and computational cost.
- **Spilling Strategy**:
  - Use chunking with overlap for tasks like summarization or question answering.
  - Truncate only when the task prioritizes the start of the text (e.g., headlines).
- **Token Counting**: Use tokenizers (e.g., Hugging Face’s `transformers`) to count tokens accurately, as character counts differ from token counts.

---

## 2. Embedding Models (Encoding Text)

### Goal
Convert text chunks into **multidimensional vectors** (embeddings) that capture semantic meaning, enabling similarity search and retrieval.

### Why It’s Important
- **Semantic Search**: Vectors represent meaning, allowing comparison of texts based on concepts, not just keywords.
- **Applications**: Finding similar documents, answering questions, clustering texts, or building recommendation systems.
- **Role in RAG**: Embeddings enable the vector store to match a query to relevant chunks.

### Steps to Build
1. **Choose an Embedding Model**:
   - **OpenAIEmbeddings** (`text-embedding-ada-002`): Paid, high-quality, 1536-dimensional vectors.
   - **HuggingFaceEmbeddings** (`sentence-transformers/all-MiniLM-L6-v2`): Free, open-source, 384-dimensional vectors.
   - **CohereEmbeddings**: Paid, good for enterprise use, ~1024 dimensions.
   - **GooglePalmEmbeddings**: Google’s embedding model, ~768 dimensions.
2. **Generate Embeddings**:
   - Pass text chunks to the model to produce vectors.
   - Each vector is a fixed-length list of numbers (e.g., 384 or 1536 dimensions).

### Token Limits and Embedding Models
- Embedding models typically process shorter inputs than LLMs, but token limits still apply:
  - **OpenAI `text-embedding-ada-002`**: 8192 tokens.
  - **Hugging Face `all-MiniLM-L6-v2`**: 512 tokens (aligned with BERT-style models).
  - **Cohere Embed**: 512 tokens (varies by version).
- If input exceeds the limit, truncate or split the text before embedding (similar to LLM preprocessing).

### Code Example
```python
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks = ["YouTube is a video platform.", "Vimeo is another video platform."]
vectors = embedding_model.embed_documents(chunks)
print(f"Vector dimensions: {len(vectors[0])}")  # 384
print(f"First vector (first 10 values): {vectors[0][:10]}")
```

### Additional Insights
- **Vector Dimensions**:
  - Higher dimensions (e.g., 1536 for `ada-002`) capture more nuanced features but require more memory.
  - Lower dimensions (e.g., 384 for `all-MiniLM-L6-v2`) are faster and lighter, suitable for local development.
- **Semantic Similarity**: Similar texts produce vectors with small angular differences, measured by **cosine similarity** (see Section 7).
- **Handling Long Inputs**: Split texts exceeding the model’s token limit using a text splitter before embedding.
- **Performance**: OpenAI’s `ada-002` excels in semantic accuracy but is costly; Hugging Face models are cost-effective for prototyping.

## Types of Embedding Generation Techniques

Embeddings are vector representations of data (text, images, etc.) used in AI tasks like semantic search, Retrieval-Augmented Generation (RAG), and chatbot development. Below are the main types of embedding generation techniques, including examples, advantages, and use cases.

### 1. Static Word Embeddings
Static embeddings assign a fixed vector to each word, regardless of context.

| **Technique** | **Description** | **Example Model** |
|---------------|-----------------|-------------------|
| **Word2Vec**  | Predicts context words given a target word (or vice versa) using skip-gram or CBOW. | Gensim Word2Vec |
| **GloVe**     | Builds a co-occurrence matrix from a corpus to learn word relationships. | Stanford GloVe |
| **FastText**  | Extends Word2Vec by incorporating subword (character n-grams) information. | Facebook FastText |

- **Pros**: Simple, fast, and computationally efficient.
- **Cons**: Context-insensitive (e.g., “bank” in "river bank" vs. "money bank" has the same vector).
- **Use Case**: Legacy NLP applications, basic text processing.

### 2. Contextual Embeddings
Contextual embeddings generate vectors based on the surrounding context, typically using transformer models.

| **Technique** | **Description** | **Example Model** |
|---------------|-----------------|-------------------|
| **BERT Embeddings** | Uses transformer layers to capture deep bidirectional context. | BERT, RoBERTa |
| **Sentence Embeddings** | Generates embeddings for entire sentences or paragraphs. | SBERT, MiniLM |
| **Token-Level Embeddings** | Produces embeddings for each token in a sentence. | BERT, GPT-style models |

- **Pros**: Highly context-aware, ideal for complex tasks.
- **Use Case**: Semantic search, chatbots, RAG pipelines, classification.

**Example** (Sentence Embeddings with SentenceTransformers):
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("Artificial Intelligence is amazing")
```

### 3. Multimodal Embeddings
Multimodal embeddings combine different data types (e.g., text and images) into a shared vector space.

| **Type** | **Description** | **Example Model** |
|----------|-----------------|-------------------|
| **CLIP** | Maps images and text into the same embedding space for cross-modal tasks. | OpenAI CLIP |
| **VisualBERT** | Combines images and text for tasks like visual question answering (VQA). | VisualBERT |
| **Image Embeddings** | Pure image vector representations from vision models. | ResNet, ViT, DINO |

- **Pros**: Enables text-image similarity and multimodal applications.
- **Use Case**: Image search, text-image matching, multimodal RAG.

### 4. Learned Embeddings (Task-Specific)
Embeddings trained as part of a model for specific tasks, such as recommendation systems or graph-based applications.

| **Use Case** | **Example** |
|--------------|-------------|
| **Recommendation Systems** | Embeddings of users/products for personalized recommendations. |
| **Graph Embeddings** | Node2Vec, DeepWalk for graph-based data. |
| **Custom Transformer Fine-Tuning** | Fine-tuned embeddings using domain-specific data. |

- **Pros**: Tailored to specific tasks, often context-aware.
- **Use Case**: Recommendation systems, graph analytics.

### 5. OpenAI Embeddings
Proprietary embeddings optimized for Azure and OpenAI ecosystems.

| **Model** | **Use** |
|-----------|---------|
| **text-embedding-ada-002** | Semantic search, RAG, chatbots (1536-dimensional vectors). |

**Example** (OpenAI Embedding):
```python
import openai
embedding = openai.Embedding.create(input=["text"], model="text-embedding-ada-002")
```

- **Pros**: High-quality, easy to integrate with Azure AI Search and RAG pipelines.
- **Use Case**: Azure-based AI agents, semantic search.

### Summary Table of Embedding Techniques

| **Type** | **Method** | **Context Aware?** | **Use Case** |
|----------|------------|--------------------|--------------|
| **Static** | Word2Vec, GloVe, FastText | ❌ No | Basic NLP, legacy apps |
| **Contextual** | BERT, RoBERTa, SBERT | ✅ Yes | RAG, semantic search, chatbots |
| **Multimodal** | CLIP, VisualBERT | ✅ Yes | Vision + language, image search |
| **Learned/Task-Specific** | Node2Vec, RecSys embeddings | ✅ Often | Recommendation, graphs |
| **Proprietary** | OpenAI Ada | ✅ Yes | Azure AI Search, RAG |

### When to Use Which Embedding?

| **Scenario** | **Recommended Embedding** |
|--------------|-----------------------|
| **Chatbot or RAG** | text-embedding-ada-002, all-MiniLM-L6-v2 |
| **Product Recommendation** | Learned embeddings, collaborative filtering |
| **Graph Data** | Node2Vec, DeepWalk |
| **Text + Image Matching** | CLIP |
| **Local Development** | SentenceTransformers (free, offline) |



## Embedding Text and Image Data in Azure

When working with both text and image data as a knowledge source (e.g., for a chatbot or RAG pipeline), the goal is to generate embeddings in a unified or aligned vector space to enable cross-modal tasks like text-to-image search or multimodal RAG. Below is a step-by-step guide tailored to your expertise in Azure AI services, Copilot Studio, and Power Automate.

### Step-by-Step Pipeline for Text and Image Embeddings

#### 1. Preprocess Text and Image Data
- **Text**: Clean text by removing HTML, normalizing case, and truncating/padding as needed.
- **Image**: Resize images (e.g., to 224x224 pixels), normalize pixel values, and convert to tensors for model compatibility.

#### 2. Choose a Multimodal or Separate Model
To enable text-image interactions, a multimodal model like CLIP is ideal because it generates embeddings in a shared vector space. Alternatively, separate models can be used for text and image embeddings if cross-modal tasks are not required.

**Option A: Multimodal Model (CLIP)**
CLIP (Contrastive Language–Image Pretraining) maps text and images into the same embedding space, enabling tasks like text-to-image or image-to-text retrieval.

**Example** (Using CLIP in Azure Machine Learning):
```python
from PIL import Image
import torch
import clip
import torchvision.transforms as transforms

# Set device (use GPU if available in Azure ML)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess image
image = preprocess(Image.open("sample.jpg")).unsqueeze(0).to(device)

# Tokenize text
text = clip.tokenize(["A dog playing in the park"]).to(device)

# Generate embeddings
with torch.no_grad():
    image_embedding = model.encode_image(image)
    text_embedding = model.encode_text(text)

# Output: image_embedding and text_embedding in the same space
print(image_embedding.shape, text_embedding.shape)  # e.g., torch.Size([1, 512])
```

**Option B: Separate Models**
If a multimodal model is unavailable or unsuitable:
- **Text**: Use SentenceTransformers (e.g., `all-MiniLM-L6-v2`) or OpenAI’s `text-embedding-ada-002` for text embeddings.
- **Image**: Use vision models like ResNet or Vision Transformer (ViT) in Azure Machine Learning.

**Example** (ResNet for Image Embeddings):
```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load and preprocess image
image = Image.open("sample_image.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
input_tensor = preprocess(image).unsqueeze(0)

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()
model.fc = torch.nn.Identity()  # Remove classification layer

# Generate embedding
with torch.no_grad():
    img_embedding = model(input_tensor)

print(img_embedding.shape)  # e.g., torch.Size([1, 2048])
```

- **Note**: Separate models create embeddings in different spaces, limiting cross-modal tasks unless aligned manually (complex).

#### 3. Store Embeddings in a Vector Database
Store text and image embeddings along with metadata in a vector database for efficient retrieval. Options include:
- **Azure AI Search**: Supports vector and hybrid search, ideal for RAG pipelines.
- **Other Vector DBs**: FAISS, Qdrant, or Redis hosted on Azure.

**Example Record Format**:
```json
{
  "id": "doc123",
  "text": "A dog playing in the park",
  "image_path": "images/dog.jpg",
  "text_vector": [0.1, 0.2, ...],
  "image_vector": [0.3, 0.4, ...]
}

---

## 3. Vector Stores (Indexing)

### Goal
Store text chunks and their embeddings in a **vector store** for efficient similarity search and retrieval.

### Why It’s Important
- Enables **fast nearest neighbor search** to find chunks most relevant to a query.
- Forms the foundation of the retrieval component in RAG.
- Supports scalability for large datasets with millions of vectors.

### Steps to Build
1. **Choose a Vector Store**:
   - **FAISS**: Lightweight, local, ideal for prototyping or small datasets.
   - **Chroma**: Persistent, simple, built-in memory store for medium-sized projects.
   - **Pinecone**: Cloud-based, scalable, suited for large-scale applications.
   - **Weaviate**: Production-grade, supports metadata filtering and hybrid search.
   - **Qdrant, Milvus**: High-performance for enterprise use, handle large volumes.
2. **Store Chunks and Embeddings**:
   - Use `.from_texts()` to index chunks and their vectors in one step.
   - Use `.add_texts()` to add new texts to an existing store.

### Code Example
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks = ["YouTube is a video platform.", "Vimeo is another video platform."]
vector_store = FAISS.from_texts(chunks, embedding_model)
```

### Additional Insights
- **FAISS**: Uses efficient indexing (e.g., HNSW, IVF) for fast search, ideal for local development.
- **Pinecone/Weaviate**: Support real-time updates and cloud scalability, but require API keys and setup.
- **Metadata**: Stores like Weaviate allow attaching metadata (e.g., document title, date) for filtering during retrieval.
- **Token Consideration**: Vector stores don’t have token limits, but the embeddings are derived from token-limited inputs.

---

## 4. Retrievers (Finding Relevant Documents)

### Goal
Retrieve the most relevant text chunks from the vector store for a given query.

### Why It’s Important
- Filters out irrelevant content, reducing input to the LLM.
- Ensures responses are fast, accurate, and cost-effective by focusing on relevant chunks.

### Steps to Build
1. **Convert Vector Store to Retriever**:
   - Use `.as_retriever()` to create a retriever from the vector store.
2. **Tune Retrieval Strategy**:
   - **search_type**:
     - `"similarity"`: Returns top-k chunks based on cosine similarity.
     - `"mmr"`: Maximum Marginal Relevance, balances relevance and diversity to avoid redundant results.
   - **search_kwargs**:
     - `k`: Number of chunks to retrieve (e.g., `k=3`).
     - `score_threshold`: Minimum similarity score for returned chunks.
3. **Retrieve Documents**:
   - Use `. keyed_relevant_documents(query)` to fetch relevant chunks.

### Code Example
```python
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("What is a video platform?")
print(f"Retrieved {len(docs)} documents: {docs}")
```

### Additional Insights
- **Retriever Types**:
  - **VectorStoreRetriever**: Standard similarity-based retrieval.
  - **MultiQueryRetriever**: Generates multiple query variations to improve coverage.
  - **ContextualCompressionRetriever**: Filters out irrelevant parts of retrieved chunks.
  - **BM25Retriever**: Keyword-based, useful for exact matches.
  - **EnsembleRetriever**: Combines vector and keyword-based retrieval.
  - **ParentDocumentRetriever**: Returns full documents instead of chunks.
  - **TimeWeightedRetriever**: Prioritizes recent documents for time-sensitive data.
- **Tuning k**: Small `k` (e.g., 1–3) for precise answers; larger `k` (e.g., 5–10) for broader context.
- **Token Limits**: Retrieved chunks must fit within the LLM’s token limit when combined with the query.

---

## 5. Large Language Models (LLMs)

### Goal
Use retrieved chunks and a user query to generate a coherent, contextually relevant response.

### Why It’s Important
- LLMs synthesize information from retrieved chunks to answer questions, summarize, or analyze content.
- Central to **Retrieval-Augmented Generation (RAG)**, combining retrieval and generation for accurate responses.

### Steps to Build
1. **Choose an LLM**:
   - **OpenAI**:
     - `gpt-3.5-turbo`: 4096 tokens, cost-effective.
     - `gpt-4`: 8192 tokens (up to 128k in some variants), high performance.
   - **Anthropic**:
     - `Claude 3`: ~200k tokens, strong reasoning.
   - **Open-Source**:
     - `Mistral-7B`: 8192 tokens, efficient for local use.
     - `LLaMA-2-7B`: 2048 tokens, research-focused.
     - `GPT4All`: Varies, optimized for local deployment.
2. **Integrate with LangChain**:
   - Use wrappers like `ChatOpenAI`, `ChatAnthropic`, or `LlamaCpp`.
   - Combine with a retrieval chain (e.g., `RetrievalQA`).
3. **Generate Response**:
   - Pass retrieved chunks and query to the LLM.

### Code Example
```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
answer = qa_chain.run("What is LangChain?")
print(answer)
```

### Additional Insights
- **Temperature**: Controls randomness (e(pm: 0 for deterministic, 1 for creative).
- **Token Limits**: Ensure query + retrieved chunks fit within the LLM’s token limit (e.g., 4096 for GPT-3.5).
- **Local Models**: Use `LlamaCpp` or `Ollama` for cost-free local LLMs, but performance may vary.
- **RAG Advantage**: Retrieved chunks provide context, reducing hallucination and improving accuracy.

---

## 6. Complete RAG Pipeline

### Overview
The RAG pipeline combines text splitting, embedding, vector storage, retrieval, and LLM generation to answer queries using large documents.

### Code Example
```python
# 1. Load and split text
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text("Your full document")

# 2. Embed
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Store in vector DB
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts(chunks, embedding)

# 4. Retrieve relevant content
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5. Use LLM
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 6. Ask question
response = qa.run("Explain YouTube ranking?")
print(response)
```

### Workflow
1. **Split**: Divide document into chunks with overlap to preserve context.
2. **Embed**: Convert chunks into multidimensional vectors.
3. **Store**: Index vectors in a vector store.
4. **Retrieve**: Find relevant chunks for a query using similarity search.
5. **Generate**: Pass chunks and query to the LLM for a response.

### Token Considerations
- Total tokens (query + chunks + output) must fit within the LLM’s limit.
- Example: For GPT-4 (8192 tokens), a 100-token query + 3 chunks of 500 tokens each (1500 tokens) leaves ~6592 tokens for the response.

---

## 7. Understanding Vectors and Multidimensional Embeddings

### What is a Vector?
A vector is an ordered list of numbers representing data. Example:
```python
[1.2, 3.4, 5.6]  # 3-dimensional vector
```

### What is Multidimensional?
A multidimensional vector has many numbers (e.g., 384, 768, 1536), capturing complex text features like:
- Words used
- Grammar
- Sentiment
- Topic
- Style
- Context
Example:
```python
[0.12, -0.45, 0.67, ..., 0.92]  # 384-dimensional vector
```

### Why Multidimensional?
- Captures nuanced semantic information.
- Example: “YouTube is a video platform” and “Vimeo is a video platform” produce similar vectors due to shared meaning.
- Higher dimensions (e.g., 1536 vs. 384) provide more detail but increase computational cost.

### Visualizing Dimensions
- **Simple Example**: A person’s features:
  ```python
  [175, 70]  # 2D: [height_cm, weight_kg]
  [175, 70, 28, 1, 0, 9]  # 6D: [height_cm, weight_kg, age, hair_color, glasses, shoe_size]
  ```
- In NLP, vectors often have 384–4096 dimensions, making them impossible to visualize but rich in information.

### Cosine Similarity
- Measures the angle between vectors to determine similarity:
  - Small angle → High similarity (cos(θ) ≈ 1).
  - Large angle → Low similarity (cos(θ) ≈ 0 or -1).
- Used in vector stores to find chunks closest to a query’s vector.

### Dimensional Sizes by Model
| **Model**                           | **Vector Size** | **Token Limit** |
|-------------------------------------|-----------------|-----------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384            | 512            |
| `sentence-transformers/all-mpnet-base-v2` | 768           | 512            |
| `OpenAI text-embedding-ada-002`     | 1536           | 8192           |
| `BERT`                              | 768            | 512            |
| `Cohere Embed`                      | ~1024          | 512            |
| GPT-style LLMs (internal embeddings) | 1024–4096      | 4096–128k      |

### Practical Code Example
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
sentence = "YouTube is a video platform."
embedding = model.encode(sentence)
print("Vector dimensions:", len(embedding))
print("Vector values (first 10):", embedding[:10])
```
**Output**:
```sql
Vector dimensions: 384
Vector values (first 10): [0.123, -0.232, 0.456, ..., 0.145]
```

---

## 8. Cheat Sheet: Tools and Options

### Text Splitters
| **Splitter**                     | **Use Case**                              | **Key Features**                              |
|----------------------------------|-------------------------------------------|-----------------------------------------------|
| `RecursiveCharacterTextSplitter` | General text (articles, books)            | Splits on characters, respects sentence boundaries |
| `TokenTextSplitter`             | Token-limited models                     | Splits based on token counts                  |
| `MarkdownHeaderTextSplitter`    | Markdown/README files                    | Splits by headers                            |
| `LanguageTextSplitter`          | Source code (Python, JS, etc.)           | Respects language syntax                     |
| `HTMLHeaderTextSplitter`        | Scraped HTML pages                       | Splits by HTML tags                          |
| `NLTKTextSplitter`, `SpacyTextSplitter` | NLP-aware splitting                | Uses linguistic rules (e.g., sentence boundaries) |

### Embedding Models
| **Model**                        | **Vector Size** | **Token Limit** | **Cost**       |
|----------------------------------|-----------------|-----------------|----------------|
| `HuggingFaceEmbeddings (all-MiniLM-L6-v2)` | 384   | 512            | Free           |
| `OpenAIEmbeddings (ada-002)`     | 1536           | 8192           | Paid           |
| `CohereEmbeddings`               | ~1024          | 512            | Paid           |
| `GooglePalmEmbeddings`           | ~768           | 512            | Paid           |

### Vector Stores
| **Store**    | **Use Case**                          | **Key Features**                          |
|--------------|---------------------------------------|-------------------------------------------|
| `FAISS`      | Local, prototyping                   | Lightweight, fast, efficient indexing      |
| `Chroma`     | Medium-sized projects, persistent     | Simple, built-in memory store             |
| `Pinecone`   | Cloud, large-scale                   | Scalable, real-time updates               |
| `Weaviate`   | Production, metadata filtering        | Hybrid search, advanced filtering         |
| `Qdrant`, `Milvus` | High-volume enterprise use     | High performance, large datasets          |

### Retrievers
| **Retriever**                    | **Use Case**                              | **Key Features**                              |
|----------------------------------|-------------------------------------------|-----------------------------------------------|
| `VectorStoreRetriever`          | Default similarity search                 | Uses cosine similarity                        |
| `MultiQueryRetriever`           | Broader query coverage                   | Generates multiple query variations          |
| `ContextualCompressionRetriever`| Remove irrelevant content                | Filters unhelpful parts of chunks             |
| `BM25Retriever`                 | Keyword-based search                     | Exact match retrieval                        |
| `EnsembleRetriever`             | Combine vector and keyword search        | Balances semantic and keyword accuracy        |
| `ParentDocumentRetriever`       | Retrieve full documents                  | Returns entire document, not just chunks      |
| `TimeWeightedRetriever`         | Time-sensitive data                     | Prioritizes recent documents                 |

### LLMs
| **Model**                        | **Token Limit** | **Cost**       | **Key Features**                       |
|----------------------------------|-----------------|----------------|----------------------------------------|
| `ChatOpenAI (gpt-3.5-turbo)`     | 4096           | Paid           | Cost-effective, fast                   |
| `ChatOpenAI (gpt-4)`             | 8192–128k      | Paid           | High performance, strong reasoning     |
| `ChatAnthropic (Claude 3)`       | ~200k          | Paid           | Excellent reasoning, large context     |
| `Mistral-7B`                     | 8192           | Free (local)   | Efficient, good for local use          |
| `LLaMA-2-7B`                     | 2048           | Free (local)   | Research-focused, high quality         |
| `GPT4All`                        | Varies         | Free (local)   | Optimized for local deployment         |

---

## 9. Summary
| **Term**                | **Meaning**                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Tokenization**        | Breaking text into tokens (words, subwords, etc.) for model processing.     |
| **Overlap**             | Shared tokens between chunks to preserve context.                           |
| **Spilling**            | Excess tokens moved to next chunk or truncated due to token limits.         |
| **Vector**              | Ordered list of numbers representing data.                                  |
| **Multidimensional**    | Vector with many values (e.g., 384–1536) capturing complex features.        |
| **Embedding**           | Vector representation of text meaning.                                      |
| **RAG Pipeline**        | Combines splitting, embedding, storage, retrieval, and LLM generation.      |

### Key Takeaways
- **Text Splitting**: Splits text into chunks with overlap to fit token limits and preserve context.
- **Embedding Models**: Convert chunks into multidimensional vectors (e.g., 384–1536 dimensions) for semantic search.
- **Vector Stores**: Index embeddings for fast retrieval (e.g., FAISS for local, Pinecone for cloud).
- **Retrievers**: Fetch relevant chunks using similarity search (e.g., cosine similarity, MMR).
- **LLMs**: Generate responses using retrieved chunks and queries, with token limits from 512 (BERT) to 200k (Claude).
- **Token Limits**: Critical for managing input size; use splitting, overlap, or truncation to handle long texts.

---

## 10. Additional Notes
- **Experimentation**: Use Hugging Face’s `transformers` library to test tokenization and embedding:
  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  tokens = tokenizer("YouTube is a video platform.", return_tensors="pt")
  print(f"Token count: {len(tokens['input_ids'][0])}")
  ```
- **Visualizing Embeddings**: Use `matplotlib` or `seaborn` to plot 2D projections of vectors via PCA or t-SNE.
- **Resources**:
  - LangChain Documentation: https://python.langchain.com/docs
  - Hugging Face Tokenizers: https://huggingface.co/docs/transformers/tokenizer_summary
  - Sentence Transformers: https://www.sbert.net/
- **Best Practices**:
  - Align `chunk_size` with model token limits (e.g., 500 for BERT’s 512).
  - Use 10–20% overlap to balance context and efficiency.
  - Choose embedding models based on trade-offs (e.g., `all-MiniLM-L6-v2` for speed, `ada-002` for accuracy).

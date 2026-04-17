# RAG Pipeline using ONLY OpenAI (No LangChain)

## 1. Imports and Setup (OpenAI API Key)

```python
import os
from openai import OpenAI

client = OpenAI(api_key="your_api_key_here")
```

Initializes OpenAI client for embeddings and chat completion.

---

## 2. Data Loading

```python
text = """
Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data.
It is widely used in healthcare, finance, and automation.
"""
```

Loads a sample document to act as knowledge base.

---

## 3. Chunking

```python
def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = chunk_text(text)
```

Splits text into smaller pieces for better retrieval.

---

## 4. Embedding

```python
def get_embedding(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

embeddings = [get_embedding(chunk) for chunk in chunks]
```

Converts chunks into vector embeddings.

---

## 5. Vector Store Creation

```python
import numpy as np

vectors = np.array(embeddings)
```

Stores embeddings in a numpy array for similarity search.

---

## 6. Retriever Setup

```python
def retrieve(query, k=2):
    query_vec = np.array(get_embedding(query))
    similarities = vectors @ query_vec
    top_k_idx = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k_idx]
```

Finds most relevant chunks using cosine similarity.

---

## 7. Prompt Augmentation and Generation

```python
def generate_answer(query):
    context = "\n".join(retrieve(query))

    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question: {query}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
```

Uses retrieved context + LLM to generate final answer.

---

# Full Runnable Script

```python
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key="your_api_key_here")

# Sample data
text = """
Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data.
It is widely used in healthcare, finance, and automation.
"""

# Chunking
def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = chunk_text(text)

# Embedding
def get_embedding(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

embeddings = [get_embedding(chunk) for chunk in chunks]
vectors = np.array(embeddings)

# Retriever
def retrieve(query, k=2):
    query_vec = np.array(get_embedding(query))
    similarities = vectors @ query_vec
    top_k_idx = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k_idx]

# Generator
def generate_answer(query):
    context = "\n".join(retrieve(query))

    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question: {query}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Example Query
query = "Where is AI used?"
answer = generate_answer(query)

print("Query:", query)
print("Answer:", answer)
```

---

## Example Query

**Input:** Where is AI used?  
**Output:** AI is used in healthcare, finance, and automation.

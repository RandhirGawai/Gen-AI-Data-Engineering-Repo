# RAG Pipeline using LangChain + OpenAI

## 1. Imports and Setup (OpenAI API Key)

```python
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

Loads required libraries and sets the OpenAI API key for authentication.

---

## 2. Data Loading

```python
text = """
Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data.
It is widely used in healthcare, finance, and automation.
"""
```

Creates a sample text document that will be used as the knowledge base.

---

## 3. Chunking

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = splitter.create_documents([text])
```

Splits the document into smaller chunks to improve retrieval accuracy.

---

## 4. Embedding

```python
embedding = OpenAIEmbeddings()
```

Converts text chunks into vector embeddings for similarity search.

---

## 5. Vector Store Creation

```python
vectorstore = FAISS.from_documents(docs, embedding)
```

Stores embeddings in FAISS for efficient retrieval.

---

## 6. Retriever Setup

```python
retriever = vectorstore.as_retriever()
```

Creates a retriever to fetch relevant chunks based on user queries.

---

## 7. Prompt Augmentation and Generation

```python
llm = ChatOpenAI()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

Combines retrieved context with LLM to generate answers.

---

# Full Runnable Script

```python
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Step 1: API Key setup
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# Step 2: Sample data
text = """
Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data.
It is widely used in healthcare, finance, and automation.
"""

# Step 3: Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = splitter.create_documents([text])

# Step 4: Embeddings
embedding = OpenAIEmbeddings()

# Step 5: Vector Store
vectorstore = FAISS.from_documents(docs, embedding)

# Step 6: Retriever
retriever = vectorstore.as_retriever()

# Step 7: RAG Chain
llm = ChatOpenAI()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Example Query
query = "Where is AI used?"
response = qa_chain.run(query)

print("Query:", query)
print("Answer:", response)
```

---

## Example Query

**Input:** Where is AI used?  
**Output:** AI is used in healthcare, finance, and automation.

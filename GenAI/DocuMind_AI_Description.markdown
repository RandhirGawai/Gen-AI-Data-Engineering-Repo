# Detailed Technical Project Description (For Documentation / Portfolio)

## DocuMind AI – Intelligent Chatbot for Documents of All Types

### Overview
This project is an end-to-end AI chatbot platform that allows users to query data from PDF documents, image files, and CSV datasets via a web interface. It integrates LangChain for data ingestion and retrieval, Groq’s LLaMA-3.1 model for conversational AI, and Kafka (Confluent Cloud) for real-time analytics streaming. The solution is containerized with Docker, deployed on AWS ECS & EC2, and optionally automated with Jenkins pipelines.

### Core Functionalities

#### Multi-Format File Ingestion & Processing
- **PDFs**: Parsed using `langchain_community.PyPDFLoader`.
- **Images**: Processed with OCR (e.g., Tesseract or AWS Rekognition) for text extraction.
- **CSV Files**: Parsed via Pandas for structured data queries.
- Extracted text is chunked with `RecursiveCharacterTextSplitter` and embedded using HuggingFace `all-MiniLM-L6-v2` into FAISS vector store.

#### Conversational AI
- Uses Groq’s `LLaMA-3.1-8b-instant` via `langchain_groq.ChatGroq`.
- Retrieval-Augmented Generation (RAG) for document-aware responses.
- Multi-turn conversation handling with session persistence in Flask.

#### Real-Time Kafka Analytics
- Publishes every message, file upload event, and bot reply to Kafka topic (`chat_events`).
- Kafka consumers process:
  - Sentiment tracking
  - Conversation quality scoring
  - Usage statistics (messages per user, daily active users, most popular intents and file types)

#### Web Application
- Flask backend with Jinja2 templating for chat UI and file uploads.
- Session-based chat history.
- Custom filters for formatting messages.

#### Error Handling & Logging
- Centralized logging with Python’s `logging` module.
- Daily log rotation and custom exceptions for better debugging.

#### Deployment & CI/CD
- Dockerized services for portability.
- **AWS ECS**: Chatbot API and Kafka consumers run as ECS tasks with auto-scaling.
- **AWS EC2**: Dedicated compute instances for heavy document/image/CSV processing.
- **AWS S3**: Storage for uploaded files.
- **Jenkins pipeline**: Automated build, push to ECR, and ECS service updates.

### Tech Stack
- **Programming Language**: Python 3.11
- **Frameworks**: Flask, LangChain
- **LLM**: Groq LLaMA-3.1-8b-instant
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Message Broker**: Apache Kafka (Confluent Cloud)
- **Libraries**: `langchain`, `langchain_community`, `langchain_groq`, `huggingface_hub`, `kafka-python`, `pandas`, `pytesseract`
- **Containerization**: Docker
- **Automation**: Jenkins Pipeline
- **Cloud Deployment**: AWS ECS, AWS EC2, AWS S3
- **Monitoring**: AWS CloudWatch

### Key Highlights
- **Multi-Format Capability**: Handles PDF, image, and CSV queries in one unified interface.
- **End-to-End RAG Architecture**: Combines document/image/CSV retrieval with generative AI for accurate, context-rich responses.
- **Scalable Real-Time Analytics**: Kafka enables independent monitoring, sentiment scoring, and usage tracking.
- **AWS Cloud Deployment**: Flexible scaling using ECS tasks and EC2 instances for compute-heavy workloads.
- **Robust & Maintainable**: Modular architecture, centralized logging, and automated CI/CD.







# GraphRAG AI – Advanced Retrieval-Augmented Generation with Graph Knowledge

### Overview
This project is an end-to-end Retrieval-Augmented Generation (RAG) platform that enables users to query complex knowledge bases through a web interface. It designs and implements RAG pipelines using LangChain and LangGraph, integrates AWS OpenSearch with KNN plugin for vector storage and retrieval, handles complex query chaining and prompt orchestration, and incorporates graph-based knowledge representations using Neo4j. The solution is containerized with Docker and deployed on AWS ECS for scalability.

### Core Functionalities

#### RAG Pipeline Design & Implementation
- Pipelines built using LangChain for data ingestion, retrieval, and generation.
- LangGraph utilized for orchestrating multi-step workflows and state management in RAG processes.

#### Vector Database Integration
- AWS OpenSearch with KNN plugin for efficient vector similarity search.
- Documents are embedded and indexed into OpenSearch for fast retrieval.

#### Complex Query Handling
- Supports query chaining where multiple retrieval and generation steps are linked.
- Prompt orchestration to dynamically construct and refine prompts based on intermediate results.

#### Graph-Based Knowledge Representations
- Neo4j used to store and query graph-structured data for enhanced context and relationships.
- Integration with RAG to combine vector search with graph traversals for more accurate responses.

#### Web Application
- Flask backend with a simple UI for user queries and displaying responses.
- Session management for multi-turn interactions.

#### Error Handling & Logging
- Centralized logging with Python’s logging module.
- Custom exceptions for pipeline failures and query errors.

#### Deployment & CI/CD
- Dockerized services for easy deployment.
- AWS ECS: RAG API and graph services run as ECS tasks with auto-scaling.
- AWS EC2: For compute-intensive tasks like embedding generation.
- AWS S3: Storage for raw data files.
- Jenkins pipeline: Automated build, test, and deployment.

### Tech Stack
- **Programming Language**: Python 3.11
- **Frameworks**: LangChain, LangGraph, Flask
- **LLM**: OpenAI GPT models (via LangChain integrations)
- **Vector Store**: AWS OpenSearch with KNN plugin
- **Graph Database**: Neo4j
- **Libraries**: langchain, langgraph, neo4j, opensearch-py, huggingface_hub
- **Containerization**: Docker
- **Automation**: Jenkins Pipeline
- **Cloud Deployment**: AWS ECS, AWS EC2, AWS S3
- **Monitoring**: AWS CloudWatch

### Key Highlights
- **Advanced RAG Pipelines**: Leverages LangChain and LangGraph for flexible, stateful RAG workflows.
- **Hybrid Retrieval**: Combines vector search in OpenSearch with graph queries in Neo4j for richer insights.
- **Complex Query Support**: Handles chained queries and orchestrated prompts for sophisticated user interactions.
- **Scalable Deployment**: AWS-based infrastructure ensures high availability and auto-scaling.
- **Robust Architecture**: Modular design with comprehensive logging and CI/CD for maintainability.

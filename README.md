
# RAG System

This repository contains the implementation of a **Retrieval-Augmented Generation (RAG)** system that combines a Neo4j graph database with large language models (LLMs) to enhance the retrieval of relevant information and provide intelligent question-answering.

## Features

- **Graph Database Integration**: Uses Neo4j to store and query data efficiently.
- **LLM Interaction**: Leverages the `Ollama` LLM for advanced language understanding and document processing.
- **Entity Extraction and Embedding Search**: Uses embeddings to retrieve relevant documents from the Neo4j graph based on user queries.
- **FastAPI-based API**: Provides a RESTful API to interact with the RAG system.

## Components

### 1. RAG.py
- **GraphDatabaseManager**: Manages interactions with the Neo4j graph database, including creating indices and running queries.
- **LLMManager**: Handles communication with the LLM for document embeddings and text generation.
- **Retriever**: Extracts entities and retrieves relevant documents using hybrid search methods.
- **ChatHandler**: Manages user queries and generates responses by interacting with the graph and the LLM.

### 2. main.py
- **FastAPI Application**: Exposes a REST API to handle question-answering requests.
  - `/question/`: Accepts a question as input and returns structured and unstructured data relevant to the query.
  
## Setup

1. **Install dependencies**:
   ```bash
   conda create -n Kgraph python==3.10
   pip install -r requirements.txt
   ```

2. **Set environment variables** for Neo4j connection:
   ```bash
   export NEO4J_URI="neo4j+s://<your-neo4j-uri>"
   export NEO4J_USERNAME="<your-username>"
   export NEO4J_PASSWORD="<your-password>"
   ```

3. **Run the FastAPI server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

- `POST /question/`: 
  - **Request**:
    - JSON Body: `{ "question": "Your question here" }`
  - **Response**:
    - `structured_data`: Extracted entities or structured information.
    - `unstructured_data`: Related documents or unstructured responses.

## Example

To ask a question:

```bash
curl -X POST "http://localhost:8000/question/" -H "Content-Type: application/json" -d '{"question": "What is the capital of France?"}'
```

You should receive a response containing relevant structured and unstructured data.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

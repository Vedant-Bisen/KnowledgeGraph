
# KnowledgeGraph RAG Setup

This repository contains a script `RAG.py` for configuring and running a Retrieval-Augmented Generation (RAG) system using `langchain`, `Neo4j`, and `OllamaEmbeddings`. The script leverages graph databases and vector stores for enhanced information retrieval and generation capabilities.

## Overview

`RAG.py` is designed to set up a RAG pipeline that integrates various components, such as:

- **Neo4j** for graph-based knowledge storage and management.
- **LangChain Community Modules** for managing embeddings, vector storage, and graph interactions.
- **Ollama** embeddings for embedding generation and retrieval.

### Key Components

- **Neo4j Configuration**: The script initializes Neo4j connection parameters using environment variables.
- **Vector Stores**: Utilizes Neo4j as a vector store for storing and retrieving embeddings.
- **LLM Integration**: Uses the Ollama embeddings for local large language model support.

## Setup

To get started with the script, make sure you have the following dependencies installed:

- Python 3.8+
- `langchain`
- `neo4j`
- `langchain_community`
- `langchain_ollama`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Vedant-Bisen/KnowledgeGraph.git
   cd KnowledgeGraph
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Update the Neo4j configuration in the script or set the environment variables:

   ```bash
   export NEO4J_URI="your-neo4j-uri"
   export NEO4J_USERNAME="your-neo4j-username"
   export NEO4J_PASSWORD="your-neo4j-password"
   ```

## Usage

Run the script using:

```bash
python RAG.py
```

This will start the RAG system, connect to the Neo4j database, and perform the necessary setup for the embeddings and knowledge graph interactions.

## Contribution

Feel free to contribute to the repository by submitting pull requests or reporting issues.

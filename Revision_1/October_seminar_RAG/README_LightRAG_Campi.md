# LightRAG Campylobacter Notebook

This repository contains a Google Colab notebook for building and querying a LightRAG/RAGAnything knowledge base from documents related to a *Campylobacter* spp. food-safety negotiation scenario. The workflow uses OpenAI models for text generation, vision-enabled document parsing, and embeddings, then queries the resulting LightRAG database to generate stakeholder descriptions for a human-in-the-loop negotiation exercise.

## Notebook

- `LightRAG_Campi_github_final.ipynb`

## Main workflow

1. **Install dependencies**
   - Installs `lightrag-hku`, `raganything`, `docling`, `openai`, `pillow`, `tqdm`, and `python-dotenv`.

2. **Load credentials**
   - Reads the OpenAI API key from Google Colab secrets via:
     - `userdata.get("api-key-MicRisk")`
   - Optionally reads `OPENAI_BASE_URL` from environment variables.

3. **Build the LightRAG database**
   - Uses documents from:
     - `/content/documents`
   - Creates/refreshes:
     - `rag_storage`
     - `output`
   - Uses:
     - text model: `gpt-4o`
     - vision model: `gpt-4o`
     - embedding model: `text-embedding-3-large`

4. **Query the database**
   - Loads the existing LightRAG database from `rag_storage`.
   - Uses the stored literature base to generate stakeholder descriptions for a discussion about food-safety microbiological criteria for *Campylobacter* spp. in broiler meat.
   - Example stakeholders include:
     - Environmental expert / food waste
     - Food Safety Authority
     - Industry
     - Consumer

5. **Optional Neo4j export and visualization**
   - Converts LightRAG GraphML output to JSON.
   - Uploads nodes and relationships to Neo4j using Cypher queries.
   - This step is marked as optional and is not directly used in the manuscript workflow.

## Requirements

The notebook is designed for Google Colab and requires:

- Python/Colab runtime
- OpenAI-compatible API access
- Documents placed in `/content/documents`
- Google Colab secrets configured for API credentials
- Optional Neo4j instance for graph inspection

Recommended Colab secrets:

```text
api-key-MicRisk
neo4j-uri-seminar
neo4j-password
```

Optional environment variable:

```text
OPENAI_BASE_URL
```

## How to run

1. Open the notebook in Google Colab.
2. Add the required API key to Colab secrets.
3. Upload source documents to `/content/documents`.
4. Run the installation and database-building cells.
5. Run the query cells to generate stakeholder descriptions.
6. Optionally run the Neo4j cells to inspect the generated knowledge graph.



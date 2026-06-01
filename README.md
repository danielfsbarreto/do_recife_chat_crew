# DoRecifeChatCrew

A RAG Q&A crew that answers questions about the **Diário Oficial do Recife** by
querying the embeddings produced by [`do_recife_embedder`](https://github.com/danielfsbarreto/do_recife_embedder)
in MongoDB Atlas Vector Search.

Two agents run sequentially:

1. **do_researcher** – runs several varied vector searches (via `MongoDBVectorSearchTool`) and consolidates the most relevant passages.
2. **do_reporter** – writes a grounded answer from those passages, cites the source (issue/date/page), and replies in the same language as the question.

## Requirements

- Python >=3.10, <3.14 and [uv](https://docs.astral.sh/uv/)
- Access to the same MongoDB Atlas cluster the embedder wrote to
- An OpenAI API key

## Setup

Install dependencies:

```bash
crewai install
```

Add the required variables to `.env`:

```bash
OPENAI_API_KEY=sk-...
MONGODB_CONNECTION_STRING=mongodb+srv://...
```

`OPENAI_API_KEY` is used both to embed the query (`text-embedding-3-large`, 3072-dim)
and to run the agents (`openai/gpt-4.1`). These must match the embedder's settings,
or vector search returns irrelevant results.

## Usage

```bash
crewai run
```

Set the `question` input in [`src/do_recife_chat_crew/main.py`](src/do_recife_chat_crew/main.py).
The final answer is printed and written to `answer.md`.

## Configuration

- [`config/agents.yaml`](src/do_recife_chat_crew/config/agents.yaml) – agent roles, goals, and LLM.
- [`config/tasks.yaml`](src/do_recife_chat_crew/config/tasks.yaml) – retrieval and answer tasks.
- [`crew.py`](src/do_recife_chat_crew/crew.py) – tool wiring (database `do-recife`, collection `do-recife-rag-enriched`, index `vector_index`).

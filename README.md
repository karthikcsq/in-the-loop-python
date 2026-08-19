# In-The-Loop Python Backend

This service exposes a LangGraph-powered planning and drafting agent over HTTP using FastAPI. It supports interactive clarification: the agent may ask follow‑up questions before producing a final result. Sessions are tracked by a thread identifier.

## Features

- Stateless HTTP API with session continuity using `thread_id`
- Two endpoints: start a run and resume after a clarification
- Compatible with the Next.js frontend via a proxy route

## Requirements

- Python 3.10+
- OpenAI API key available to the backend process

## Installation

Using uv (recommended):

```powershell
uv add fastapi uvicorn
```

Or using pip inside a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn
```

The project code depends on LangGraph and LangChain packages declared in `pyproject.toml`. If you are not using `uv`, install them with:

```powershell
pip install langgraph langchain-openai python-dotenv
```

## Configuration

Set the following environment variables for the backend process:

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, default: `gpt-4o-mini`)
- `OPENAI_TEMPERATURE` (optional, default: `0.7`)

You can load them from a `.env` file via `python-dotenv` if desired.

## Running the API

From the `in-the-loop-python` directory:

```powershell
uvx uvicorn graph_api:app --reload --port 8000
```

Alternative:

```powershell
uv run uvicorn graph_api:app --reload --port 8000
```

Ensure the working directory is the project folder so the module name `graph_api` resolves correctly.

## API Reference

Base URL: `http://127.0.0.1:8000`

### POST /start

Start a new agent run for a given `thread_id`.

Request body:

```json
{
    "thread_id": "string",
    "essay_prompt": "string",
    "task_type": "essay" | "code" | null
}
```

Response (one of):

- Interrupt pending clarification

```json
{
    "type": "interrupt",
    "query": "string",
    "options": ["string", "..."]
}
```

- Final result

```json
{
    "type": "final",
    "draft": "string"
}
```

### POST /resume

Resume a run after providing an answer to a clarification.

Request body:

```json
{
    "thread_id": "string",
    "value": "string"
}
```

Response matches `/start` (either another interrupt or the final result).

## Session Model

- Provide a stable `thread_id` for each conversation from the client.
- The backend stores intermediate state using an in‑memory checkpointer (see `langgraph_model.build_app`).
- Multiple clarify‑answer cycles are supported until a final draft is produced.

## Troubleshooting

- “Could not import module”: run from the `in-the-loop-python` folder and use `graph_api:app`.
- 401/403 from model calls: verify `OPENAI_API_KEY` in the backend environment.
- CORS issues: `graph_api.py` enables permissive CORS for development. Restrict in production.
   OPENAI_API_KEY=your_api_key_here


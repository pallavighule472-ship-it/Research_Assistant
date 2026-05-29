# AI Research Assistant

An autonomous multi-agent research platform that takes a topic, searches the web, extracts and synthesizes information across multiple iterations, and produces a cited long-form report — streamed live to a browser UI.

---

## Architecture

```
┌─────────────────────┐        SSE Stream        ┌──────────────────────┐
│  Streamlit Frontend │ ◄──────────────────────── │  FastAPI Backend     │
│  Research_Frontend  │                           │  Research_Backend    │
│  :8501              │ ──── POST /research/stream ►  _Server.py  :8000  │
└─────────────────────┘                           └──────────┬───────────┘
                                                             │
                                                   LangGraph StateGraph
                                                             │
                          ┌──────────────────────────────────▼──────────────────────┐
                          │                    Research_Backend.py                   │
                          │                                                          │
                          │  user_query → web_search → extract_info → summarize     │
                          │       ↑                                        │         │
                          │       └──────── evaluate ◄─────────────────── ┘         │
                          │                    │                                     │
                          │              research_writer                             │
                          └──────────────────────────────────────────────────────────┘
```

**Key design choices:**
- LangGraph `StateGraph` manages the iterative research loop (up to 5 iterations)
- FastAPI streams node events and LLM tokens to the frontend via SSE
- All heavy I/O (search, crawl, extract, write) runs in `ThreadPoolExecutor` pools
- Token-level streaming: each GPT-4o token is forwarded to the UI via `BaseCallbackHandler`

---

## Features

- Multi-source search: DuckDuckGo, Wikipedia, Tavily (optional)
- Parallel web crawling and fact extraction across search results
- Iterative evaluation loop — the agent self-assesses and re-searches if needed
- Adaptive report format: academic abstract, practical guide, how-to, or comparison — based on the query
- Topic-relevant images via the Wikipedia REST API
- Multilingual output (English, Hindi, Spanish, French, German, Chinese)
- Configurable writing style and report length
- Live SSE streaming of node progress and LLM tokens
- Export to Markdown, plain text, and PDF
- LangSmith tracing (optional, auto-enabled when `LANGCHAIN_API_KEY` is set)
- 37-test pytest suite

---

## Project Structure

```
Research_Assistant/
├── Research_Backend.py         # LangGraph graph + all agent nodes
├── Research_Backend_Server.py  # FastAPI SSE server
├── Research_Frontend.py        # Streamlit UI
├── run.py                      # Launcher — starts both services
├── test_research.py            # pytest test suite
├── Requirements.txt
└── .env                        # API keys (not committed)
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <repo-url>
cd Research_Assistant
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

**2. Install dependencies**

```bash
pip install -r Requirements.txt
```

**3. Configure API keys**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...

# Optional — enables Tavily search (higher quality results)
TAVILY_API_KEY=tvly-...

# Optional — enables LangSmith tracing
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=Research-Assistant
```

---

## Running

```bash
python run.py
```

This starts the FastAPI backend on `http://localhost:8000` and the Streamlit frontend on `http://localhost:8501`, then opens the browser automatically.

To run services separately:

```bash
# Backend only
uvicorn Research_Backend_Server:server --host 127.0.0.1 --port 8000

# Frontend only
streamlit run Research_Frontend.py
```

---

## Running Tests

```bash
pytest test_research.py -v
```

No API keys are required — all external calls are mocked.

---

## Agent Pipeline

| Node | What it does |
|---|---|
| `user_query` | Generates a 5-7 step research plan (section titles + search queries) via GPT-4o-mini |
| `web_search` | Runs DuckDuckGo, Wikipedia, and Tavily in parallel; crawls discovered URLs |
| `extract_info` | Extracts structured facts from each source in parallel via GPT-4o-mini |
| `summarize_notes` | Compresses and deduplicates all note blocks via GPT-4o-mini |
| `evaluate` | Scores research against 6 quality criteria; loops back or proceeds via GPT-4o-mini |
| `optimize_research` | Identifies research gaps and generates targeted new queries via GPT-4o-mini |
| `refine_query` | Sharpens gap-fill queries for better search results via GPT-4o-mini |
| `research_writer` | Writes opening, all body sections, and closing concurrently via GPT-4o |
| `save_notes` | Logs a completion summary (iteration count, sources crawled, knowledge base size) |

---

## Models Used

| Model | Used for |
|---|---|
| `gpt-4o-mini` | Planning, extraction, summarization, evaluation (all intermediate nodes) |
| `gpt-4o` | Final report writing only (opening, sections, closing) |

---

## Docker

Build once, run both services:

```bash
docker compose up --build
```

- Streamlit UI → http://localhost:8501
- FastAPI docs → http://localhost:8000/docs

The frontend service waits for the backend health check to pass before starting (`condition: service_healthy`).

```bash
# Stop
docker compose down
```

> Make sure your `.env` file with API keys exists in the project root before running.

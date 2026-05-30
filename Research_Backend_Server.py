import json
import asyncio
import os
from fastapi import FastAPI, Security, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

load_dotenv()

from Research_Backend import app as graph_app

_APP_PASSWORD = os.getenv("APP_PASSWORD", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _verify(api_key: str = Security(_api_key_header)):
    if _APP_PASSWORD and api_key != _APP_PASSWORD:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid password")


class TokenQueueHandler(BaseCallbackHandler):
    """Forwards each LLM token to the SSE queue on the event loop thread."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._queue = queue
        self._loop = loop

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        payload = json.dumps({"type": "token", "text": token}, ensure_ascii=False)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, f"data: {payload}\n\n")

server = FastAPI(
    title="Research Assistant Backend",
    description="Decoupled LangGraph microservice for dynamic academic research"
)

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    target_language: str = "English"
    writing_style: str = "Book-Style Detailed"
    target_length: str = "Medium-depth (3-4 paragraphs)"
    max_iteration: int = Field(default=3, ge=1, le=5)


@server.get("/")
def root():
    return RedirectResponse(url="/docs")


@server.get("/health")
def health():
    return {"status": "healthy", "service": "Research Assistant Backend"}


@server.post("/research/stream", dependencies=[Depends(_verify)])
async def stream_research(payload: ResearchRequest):
    """SSE endpoint — streams LangGraph node events to the frontend in real-time."""
    initial_state = {
        "user_query":      payload.query,
        "target_language": payload.target_language,
        "writing_style":   payload.writing_style,
        "target_length":   payload.target_length,
        "iteration":       0,
        "max_iteration":   payload.max_iteration,
        "web_results":     "",
        "web_links":       [],
        "final_answer":    "",
        "plan":            [],
        "notes":           [],
        "is_complete":     False,
    }

    async def sse_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def run_graph():
            handler = TokenQueueHandler(queue, loop)
            config = RunnableConfig(callbacks=[handler])
            try:
                for event in graph_app.stream(initial_state, config=config, stream_mode="updates"):
                    for node_name, state_update in event.items():
                        chunk = {"node": node_name, "state": state_update}
                        loop.call_soon_threadsafe(
                            queue.put_nowait, f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        )
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, f"data: {json.dumps({'error': str(e)})}\n\n"
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, run_graph)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

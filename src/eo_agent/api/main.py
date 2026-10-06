"""FastAPI app.

    uvicorn eo_agent.api.main:app --port 8000

A tool failure is not an HTTP error: the user got an answer, so it is a 200.
Only unexpected failures return 5xx (503 if the language model itself is unreachable).
"""
import logging
from contextlib import asynccontextmanager
from typing import Any

import openai
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

from eo_agent.agent.runtime import AgentRuntime
from eo_agent.config import LANGUAGE_NAMES, PLACE_LANGUAGES
from eo_agent.observability.tracing import SAFE_SESSION_ID, setup_logging

log = logging.getLogger("eo_agent.api")
SESSION_ID_PATTERN = SAFE_SESSION_ID.pattern


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN, description="Letters, digits, _ . - (max 64)")
    message: str = Field(min_length=1, max_length=4000)
    language: str | None = Field(
        None, pattern="^(" + "|".join(PLACE_LANGUAGES) + ")$",
        description="Language in which place names are searched (en, it, es, fr, de). Optional: the session keeps its last setting, English at first.",
    )


class ToolCallInfo(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    turn: int
    tool_calls: list[ToolCallInfo]


def create_app(runtime: AgentRuntime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        app.state.runtime = runtime or AgentRuntime()
        try:
            await app.state.runtime.ensure_graph()  # the MCP client and the graph are created once, here
        except Exception:
            log.exception("startup could not prepare the agent; it will be retried on the first request")
        if not app.state.runtime.mcp_connected:
            log.warning("started without the MCP server: chats get a clear reply until it is reachable")
        yield

    app = FastAPI(title="EO scene agent", lifespan=lifespan)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            result = await app.state.runtime.chat(request.session_id, request.message, request.language)
        except openai.OpenAIError as exc:  # the language model could not be reached or refused
            log.error("language model call failed: %s: %s", type(exc).__name__, exc)
            raise HTTPException(503, "The language model is not available right now. Try again in a moment.") from None
        except Exception:
            log.exception("unexpected failure while handling the chat request")
            raise HTTPException(500, "Internal error. The details are in the server log and in the session trace.") from None
        return ChatResponse(**result.__dict__)

    @app.get("/traces/{session_id}")
    async def get_trace(session_id: str = Path(pattern=SESSION_ID_PATTERN)) -> dict[str, Any]:
        events = app.state.runtime.trace(session_id)
        if not events:
            raise HTTPException(404, "No trace for this session.")
        return {"session_id": session_id, "events": events}

    @app.get("/sessions/{session_id}/memory")
    async def get_memory(session_id: str = Path(pattern=SESSION_ID_PATTERN)) -> dict[str, Any]:
        info = await app.state.runtime.session_info(session_id)
        if info is None:
            raise HTTPException(404, "No such session.")
        return info

    @app.get("/health")
    async def health() -> dict[str, Any]:
        rt = app.state.runtime
        s = rt.settings
        return {
            "status": "ok", "mcp_connected": rt.mcp_connected, "mcp_error": rt.mcp_error, "model": s.llm_model,
            "place_languages": {code: LANGUAGE_NAMES[code] for code in PLACE_LANGUAGES},
            "context_policy": {"max_window": s.max_window, "summary_trigger": s.summary_trigger,
                               "max_tool_chars": s.max_tool_chars, "max_steps": s.max_steps},
        }

    return app


app = create_app()

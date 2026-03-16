import os
import time
import uuid
from typing import Dict, Optional

from fastapi import Cookie, FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import HomeworkAgent


SESSION_TTL_SECONDS = 60 * 60 * 6  # 6 hours


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    model: Optional[str] = Field(default=None, description="Optional model override")


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class NewChatRequest(BaseModel):
    model: Optional[str] = None


class NewChatResponse(BaseModel):
    session_id: str


class SessionState:
    def __init__(self, agent: HomeworkAgent, created_at: float):
        self.agent = agent
        self.created_at = created_at
        self.last_used_at = created_at


app = FastAPI(title="SmartTutor API", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

_sessions: Dict[str, SessionState] = {}


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s.last_used_at > SESSION_TTL_SECONDS]
    for sid in expired:
        _sessions.pop(sid, None)


def _create_session(model: Optional[str]) -> str:
    model_id = (model or os.getenv("DASHSCOPE_MODEL") or "deepseek-r1").strip() or "deepseek-r1"
    agent = HomeworkAgent(model=model_id)
    sid = uuid.uuid4().hex
    _sessions[sid] = SessionState(agent=agent, created_at=time.time())
    return sid


def _get_or_create_session(session_id: Optional[str], model: Optional[str]) -> str:
    _cleanup_sessions()
    if session_id and session_id in _sessions:
        _sessions[session_id].last_used_at = time.time()
        return session_id
    return _create_session(model=model)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/new_chat", response_model=NewChatResponse)
def new_chat(req: NewChatRequest, response: Response) -> NewChatResponse:
    sid = _create_session(model=req.model)
    response.set_cookie("session_id", sid, httponly=True, samesite="lax")
    return NewChatResponse(session_id=sid)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, response: Response, session_id: Optional[str] = Cookie(default=None)) -> ChatResponse:
    sid = _get_or_create_session(session_id=session_id, model=req.model)
    response.set_cookie("session_id", sid, httponly=True, samesite="lax")
    state = _sessions[sid]
    answer = state.agent.ask(req.message, stream=False)
    state.last_used_at = time.time()
    return ChatResponse(session_id=sid, answer=answer)


@app.post("/api/summarize", response_model=ChatResponse)
def summarize(response: Response, session_id: Optional[str] = Cookie(default=None)) -> ChatResponse:
    sid = _get_or_create_session(session_id=session_id, model=None)
    response.set_cookie("session_id", sid, httponly=True, samesite="lax")
    state = _sessions[sid]
    answer = state.agent.summarize()
    state.last_used_at = time.time()
    return ChatResponse(session_id=sid, answer=answer)


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from langgraph_model import build_app

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_app()

class StartRequest(BaseModel):
    thread_id: str
    essay_prompt: str
    task_type: str | None = None

class ResumeRequest(BaseModel):
    thread_id: str
    value: str

def serialize_result(result: dict):
    if isinstance(result, dict) and "__interrupt__" in result:
        payload = result["__interrupt__"][0].value or {}
        return {"type": "interrupt", "query": payload.get("query"), "options": payload.get("options")}
    if isinstance(result, dict) and "draft" in result:
        return {"type": "final", "draft": result["draft"]}
    return {"type": "error", "error": "Unexpected result from graph"}

@app.post("/start")
def start(req: StartRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    init = {"essay_prompt": req.essay_prompt}
    if req.task_type:
        init["task_type"] = req.task_type
    result = graph.invoke(init, config=config)
    return serialize_result(result)

@app.post("/resume")
def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke(Command(resume=req.value), config=config)
    return serialize_result(result)

# Run: uvicorn server:app --reload --port 8000
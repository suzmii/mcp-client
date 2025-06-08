import os
from dotenv import load_dotenv

load_dotenv()

import uvicorn

import asyncio
from typing import Dict, List
from fastapi import FastAPI, Request, HTTPException
from mcp_client import MCPClient


sessions: Dict[str, List[Dict[str, str]]] = {}
client: MCPClient = None

app = FastAPI(debug=True)

@app.on_event("startup")
async def startup():
    global client
    base_url = os.getenv("BASE_URL")
    token = os.getenv("TOKEN") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")
    if not token:
        raise RuntimeError("Missing TOKEN or OPENAI_API_KEY in environment")
    client = await MCPClient.create(
        base_url=base_url,
        token=token,
        model=model
    )

@app.post("/sessions")
async def create_session(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    context = data.get("context", [])
    sessions[session_id] = context.copy()
    return {"success": True}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    content = data.get("content")
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = sessions[session_id]
    messages.append({"role": "user", "content": content})
    try:
        response = await client.process_query(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    messages.append({"role": "assistant", "content": response})
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
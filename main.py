import asyncio
import os
from typing import Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from mcp_client import MCPClient

load_dotenv()


sessions: Dict[str, List[Dict[str, str]]] = {}
client: MCPClient = None

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global client
    base_url = os.getenv("BASE_URL")
    token = os.getenv("TOKEN") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")
    if not token:
        raise RuntimeError("Missing TOKEN or OPENAI_API_KEY in environment")
    client = await MCPClient.create(base_url=base_url, token=token, model=model)


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


@app.post("/nosession-chat")
async def nosession_chat(request: Request):
    data = await request.json()
    content = data.get("content")
    try:
        response = await client.process_query([{"role": "user", "content": f"以下是用户的问题，请注意你不需要回答用户，而是简要概括用户含义，返回十个字以内的内容作为概述：{content}{content}"}])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=0)

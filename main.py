import asyncio
import os
from typing import Dict, List

import uvicorn
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from mcp_client import MCPClient

load_dotenv()


sessions: Dict[str, List[Dict[str, str]]] = {}
client: MCPClient = None

# Token验证服务配置
# 验证token的接口URL，暂时留空
TOKEN_VERIFY_URL = "http://localhost:8090/v1/user/token/verify/access"


class TokenVerifyMiddleware(BaseHTTPMiddleware):
    """Token验证中间件 - 从Headers获取token并调用外部服务验证"""

    async def dispatch(self, request: Request, call_next):
        # 从请求头中获取token
        access_token = self.extract_token_from_headers(request)

        if access_token:
            # 调用外部服务验证token
            is_valid = await self.verify_token_with_service(access_token)
            if not is_valid:
                return Response(
                    content='{"detail": "Token验证失败"}',
                    status_code=401,
                    media_type="application/json"
                )
        else:
            return Response(
                content='{"detail": "缺少访问令牌"}',
                status_code=401,
                media_type="application/json"
            )

        response = await call_next(request)
        return response

    def extract_token_from_headers(self, request: Request) -> str:
        """
        从请求头中提取token
        支持多种常见的token传递方式
        """
        # 方式1: Authorization Bearer token
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            return authorization[7:]  # 去掉 "Bearer " 前缀

        # 方式2: 自定义 Access-Token 头
        access_token = request.headers.get("Access-Token")
        if access_token:
            return access_token

        # 方式3: 自定义 X-Access-Token 头
        x_access_token = request.headers.get("X-Access-Token")
        if x_access_token:
            return x_access_token

        return None

    async def verify_token_with_service(self, access_token: str) -> bool:
        """
        使用POST方法调用外部服务验证token

        Args:
            access_token: 需要验证的token

        Returns:
            bool: 验证成功返回True，失败返回False
        """
        if not TOKEN_VERIFY_URL:
            # 如果验证URL为空，暂时返回True（开发阶段）
            print(f"警告: TOKEN_VERIFY_URL为空，跳过token验证。Token: {access_token}")
            return True

        try:
            async with httpx.AsyncClient() as client:
                # 使用POST方法调用验证服务，携带JSON参数
                response = await client.post(
                    TOKEN_VERIFY_URL,
                    json={"access_token": access_token},
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                # 通过状态码判断验证是否成功
                return response.status_code == 200
        except httpx.TimeoutException:
            print(f"Token验证请求超时: {TOKEN_VERIFY_URL}")
            return False
        except httpx.RequestError as e:
            print(f"Token验证请求失败: {e}")
            return False
        except Exception as e:
            print(f"Token验证过程中发生未知错误: {e}")
            return False


app = FastAPI(debug=True)

# 添加Token验证中间件（在CORS中间件之前）
app.add_middleware(TokenVerifyMiddleware)

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
        response = await client.process_query([{"role": "user", "content": f"以下是用户的问题，你不需要回答用户，你需要简要概括用户含义，返回十个字以内的内容作为概述：{content}"}])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=0)

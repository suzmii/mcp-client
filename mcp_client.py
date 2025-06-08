import asyncio
import json
import os
from asyncio import Task
from contextlib import AsyncExitStack
from typing import Dict, List, Optional, Tuple

import httpx
from mcp import ClientSession, StdioServerParameters, ListToolsResult
from mcp.client.stdio import stdio_client
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageParam
from openai.types.shared_params import FunctionDefinition


class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai: Optional[OpenAI] = None
        self.model_name: Optional[str] = None

        self.sessions: Dict[str, ClientSession] = {}
        self.tool_name_to_server_map: Dict[str, str] = {}
        self.mcp_methods: List[ChatCompletionToolParam] = []
        self.max_token = 1024

    @classmethod
    async def create(cls, base_url: str, token: str, model: str) -> "MCPClient":
        """
        创建客户端时仅初始化 OpenAI，不加载本地 MCP Server，避免跨平台子进程错误。
        """
        self = cls()
        self.openai = OpenAI(base_url=base_url, api_key=token)
        self.model_name = model
        # 本地 MCP Server 加载已禁用以避免启动失败
        await self.load_and_connect_servers()
        return self

    async def load_and_connect_servers(self):
        try:
            with open("mcp_servers.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            servers = data.get("mcpServers", {})
            enabled_servers = {name: cfg for name,
                               cfg in servers.items() if cfg.get("enable")}

            if not enabled_servers:
                raise ValueError("配置中没有启用的 MCP Server")

            wait_list: List[Task[Tuple[str,
                                       ClientSession, ListToolsResult]]] = []

            for name, config in enabled_servers.items():
                async def load(name, _config):
                    server_params = StdioServerParameters(
                        command=_config["command"],
                        args=_config["args"],
                        env=None
                    )
                    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                    stdio, write = stdio_transport
                    _session = await self.exit_stack.enter_async_context(
                        ClientSession(stdio, write))
                    await _session.initialize()
                    _tool_list = await _session.list_tools()
                    return name, _session, _tool_list

                wait_list.append(asyncio.create_task(load(name, config)))

            for task in wait_list:
                name, session, tool_list = await task
                self.sessions[name] = session
                for tool in tool_list.tools:
                    self.tool_name_to_server_map[tool.name] = name
                    self.mcp_methods.append(ChatCompletionToolParam(
                        type="function",
                        function=FunctionDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.inputSchema
                        )
                    ))

        except Exception as e:
            await self.cleanup()
            raise e

    async def process_query(self, messages: List[ChatCompletionMessageParam]) -> str:
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.mcp_methods,
            tool_choice="auto",
            max_tokens=self.max_token
        )

        final_text = []
        current_messages = messages.copy()

        while True:
            choice = response.choices[0]
            message = choice.message

            if message.content:
                final_text.append(message.content.strip())

            if not getattr(message, "tool_calls", None):
                break

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                server_name = self.tool_name_to_server_map.get(tool_name)
                if not server_name:
                    raise ValueError(f"未找到工具 {tool_name} 对应的 MCP Server")

                session = self.sessions[server_name]
                result = await session.call_tool(tool_name, tool_args)

                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                }
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content)
                }

                current_messages.append(assistant_message)
                current_messages.append(tool_message)

                response = self.openai.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    tools=self.mcp_methods,
                    tool_choice="auto",
                    max_tokens=self.max_token
                )

        return "\n".join(final_text)

    async def cleanup(self):
        await self.exit_stack.aclose()


async def select_ollama_model(base_url: str) -> str:
    if ":11434" not in base_url:
        raise ValueError("Ollama 服务地址无效")

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url.replace('/v1', '')}/api/tags")
        response.raise_for_status()
        models_data = response.json()

    models = [model["name"] for model in models_data.get("models", [])]
    if not models:
        raise ValueError("没有检测到本地 Ollama 模型")

    return models[0]  # 默认返回第一个模型

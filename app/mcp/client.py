"""MCP client wrapper: launches app/mcp/server.py as a subprocess and speaks
MCP over stdio, exposing a single async `call_tool` used by the chat
assistant and the report generator.

The subprocess + session are started lazily on first use and kept open for
the lifetime of this client instance -- restarting the subprocess per call
would also throw away the server's synced ClauseIndex (see server.py's
module docstring), so the session is a long-lived resource, not a
per-request one.
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolError(RuntimeError):
    """Raised when an MCP tool call returns an error result."""


class MCPToolClient:
    def __init__(
        self, command: list[str] | None = None, env: dict[str, str] | None = None
    ) -> None:
        self._command = command or [sys.executable, "-m", "app.mcp.server"]
        self._env = env
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        if self._session is not None:
            return
        stack = AsyncExitStack()
        try:
            params = StdioServerParameters(
                command=self._command[0], args=self._command[1:], env=self._env
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            # If anything after the stack is created raises -- including a
            # failed handshake in initialize() -- close whatever was already
            # entered so the subprocess doesn't leak; nothing outside this
            # method holds a reference to `stack` once the exception unwinds.
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def call_tool(self, name: str, **kwargs) -> dict:
        if self._session is None:
            await self.start()
        assert self._session is not None  # for type-checkers; start() sets it

        result = await self._session.call_tool(name, arguments=kwargs)
        text = result.content[0].text if result.content else "{}"
        if result.isError:
            raise MCPToolError(text)
        if result.structuredContent is not None:
            return result.structuredContent
        import json

        return json.loads(text)

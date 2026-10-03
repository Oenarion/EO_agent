"""One HTTP helper for both external APIs: custom User-Agent, timeout, one retry."""
import asyncio
import logging

import httpx

from eo_agent.config import USER_AGENT
from eo_agent.mcp_server.schemas import ToolFailure

log = logging.getLogger("eo_agent.mcp")

TIMEOUT_S = 15.0
RETRY_DELAY_S = 0.5  # tests set this to 0


async def send(service: str, method: str, url: str, **kwargs) -> httpx.Response:
    """Send a request. Retry once on timeout, connection error or 5xx.

    Returns the response for anything below 500 (the caller handles 404, 400).
    Raises ToolFailure(upstream_error) if both attempts fail.
    """
    reason = ""
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT}) as client:
                response = await client.request(method, url, **kwargs)
            if response.status_code < 500:
                return response
            reason = f"HTTP {response.status_code}"
        except httpx.TransportError as exc:  # includes timeouts and connection errors
            reason = type(exc).__name__
        log.warning("%s call failed (attempt %d): %s", service, attempt, reason)
        if attempt == 1:
            await asyncio.sleep(RETRY_DELAY_S)
    raise ToolFailure("upstream_error", f"{service} is not responding ({reason}) after 2 attempts.")

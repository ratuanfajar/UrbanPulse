"""Web search retrieval via the Tavily Search API.

This replaces the document/vector-store retrieval step of a classic RAG pipeline:
Tavily performs the "find the most relevant information" job, so no local embeddings
or vector database are required.
"""

import logging
import httpx
from chatbot.config import get_settings

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Raised when the web search provider cannot be reached or is misconfigured."""


class WebSearchService:
    """Thin wrapper around the Tavily Search API."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def search(self, query: str) -> list[dict]:
        """
        Run a web search for *query*.

        Returns a list of result dicts: {title, url, snippet, score}.
        """
        settings = self._settings
        if not settings.search_ready:
            raise SearchError(
                "Web search is not configured. Set TAVILY_API_KEY in the environment."
            )

        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": settings.search_depth,
            "max_results": settings.search_max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(settings.tavily_base_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Tavily returned HTTP %s: %s", exc.response.status_code, exc.response.text)
            raise SearchError(
                f"Web search provider returned an error (HTTP {exc.response.status_code}). "
                "Check that TAVILY_API_KEY is valid."
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("Tavily request failed: %s", exc)
            raise SearchError("Could not reach the web search provider.") from exc

        results: list[dict] = []
        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title", "").strip() or item.get("url", ""),
                    "url": item.get("url", ""),
                    "snippet": (item.get("content") or "").strip(),
                    "score": item.get("score"),
                }
            )

        logger.info("Web search for %r returned %d results.", query, len(results))
        return results

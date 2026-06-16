"""RAG pipeline (web-search variant):

question -> web search (retrieval) -> build context -> OpenRouter LLM -> grounded English answer.

There is intentionally no embedding / vector-store step: the web search provider performs the
retrieval, so the answer is grounded in fresh information from the open web rather than a fixed
local document set.
"""

import logging

from openai import OpenAI

from chatbot.context_store import get_recent, get_summary, save_message, save_summary
from chatbot.config import get_settings
from chatbot.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from chatbot.response_formatter import format_to_n_sentences
from chatbot.summary_service import generate_summary_from_messages
from chatbot.search_service import SearchError, WebSearchService

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates web-search retrieval + OpenRouter answer generation."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._search = WebSearchService()
        self._client: OpenAI | None = None

    # ── lifecycle ────────────────────────────────────────────────────
    def initialize(self) -> None:
        settings = self._settings
        if not settings.llm_ready:
            logger.warning(
                "OPENROUTER_API_KEY is not set. The chatbot will return a clear error "
                "until it is configured."
            )
            return
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers={
                key: value
                for key, value in {
                    "HTTP-Referer": settings.openrouter_site_url,
                    "X-Title": settings.openrouter_app_title,
                }.items()
                if value
            },
        )
        logger.info("OpenRouter client ready (model=%s).", settings.openrouter_model)

    @property
    def llm_ready(self) -> bool:
        return self._client is not None

    @property
    def search_ready(self) -> bool:
        return self._settings.search_ready

    # ── main entry point ─────────────────────────────────────────────
    async def ask(self, question: str, session_id: str = "default") -> dict:
        """
        Run the full web-search RAG pipeline for *question*.

        Returns dict with keys: answer (str), sources (list[dict]).
        """
        if self._client is None:
            raise RuntimeError(
                "LLM is not configured. Set OPENROUTER_API_KEY in the environment."
            )

        save_message(session_id, "user", question)

        recent = get_recent(session_id, limit=10)
        summary = get_summary(session_id)
        if not self._is_relevant_to_slum_context(question, summary, recent):
            answer = (
                "This question is outside the scope of the previous slum/non-slum discussion. If you'd like to continue, please ask a related question or start a new session"
            )
            save_message(session_id, "assistant", answer)
            return {"answer": answer, "sources": []}

        # 1. Retrieve from the web.
        results = await self._search.search(question)

        # 2. Build the context block from the retrieved snippets.
        context = self._format_context(results)
        memory = self._format_memory(summary, recent)

        # 3. Ask the LLM, grounded in the retrieved context.
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            memory=memory,
            question=question,
        )
        settings = self._settings

        completion = self._client.chat.completions.create(
            model=settings.openrouter_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = format_to_n_sentences(completion.choices[0].message.content or "", n=2)

        sources = [
            {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
            for r in results
            if r.get("url")
        ]
        save_message(session_id, "assistant", answer)
        save_summary(
            session_id,
            generate_summary_from_messages([text for _, text in get_recent(session_id, limit=8)]),
        )
        return {"answer": answer.strip(), "sources": sources}

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _format_context(results: list[dict]) -> str:
        if not results:
            return "(No web search results were found for this query.)"
        blocks = []
        for i, r in enumerate(results, start=1):
            blocks.append(
                f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _format_memory(summary: str, recent: list[tuple[str, str]]) -> str:
        if not summary and not recent:
            return "(No prior conversation memory.)"
        recent_text = "\n".join(f"{role}: {text}" for role, text in recent[-4:])
        return f"Summary: {summary or '(none)'}\nRecent turns:\n{recent_text}"

    @staticmethod
    def _is_relevant_to_slum_context(
        question: str, summary: str, recent: list[tuple[str, str]]
    ) -> bool:
        q = question.lower()
        domain_keywords = [
            "slum",
            "kumuh",
            "permukiman",
            "settlement",
            "kampung",
            "bekasi",
            "drainage",
            "sanitation",
            "air bersih",
            "infrastructure",
            "infrastruktur",
            "non slum",
            "non-slum",
        ]
        followup_markers = [
            "itu",
            "tersebut",
            "lanjut",
            "follow up",
            "how about",
            "what about",
            "bagaimana dengan",
            "lalu",
            "selanjutnya",
            "lebih lanjut",
        ]
        if any(keyword in q for keyword in domain_keywords):
            return True
        if summary and any(marker in q for marker in followup_markers):
            return True
        return False


# Re-export so route handlers can catch a single error type.
__all__ = ["RAGService", "SearchError"]

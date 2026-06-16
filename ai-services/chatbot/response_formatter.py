import re

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def format_to_n_sentences(text: str, n: int = 2) -> str:
    """Return at most n sentences from text."""
    cleaned = " ".join(text.split()) if text else ""
    sentences = _SENTENCE_SPLIT_RE.split(cleaned) if cleaned else []
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= n:
        return " ".join(sentences)
    return " ".join(sentences[:n])

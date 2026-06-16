import re
from typing import List

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def generate_summary_from_messages(messages: List[str], max_sentences: int = 3) -> str:
    """Compact conversation summary for follow-up turns."""
    text = " ".join(messages).replace("\n", " ").strip()
    sentences = _SENTENCE_SPLIT_RE.split(text.strip()) if text else []
    selected = [s.strip() for s in sentences if s.strip()][:max_sentences]
    return " ".join(selected)

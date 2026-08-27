"""Content guard - exact whole-word match against the corpus profanity list.

This is the one piece of the old `evaluate.py` worth keeping verbatim: set-based
O(1) membership after lowercasing and stripping punctuation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_PUNCT = re.compile(r"[^\w\s]")


def contains_inappropriate_content(text: str, inappropriate_words: Iterable[str]) -> bool:
    words = set(_PUNCT.sub(" ", text.lower()).split())
    return bool(words & set(inappropriate_words))

"""One pass over the answer with spaCy, reused by every metric."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

HEDGE_PATTERNS = [
    r"\bi think\b", r"\bi guess\b", r"\bi believe\b", r"\bmaybe\b", r"\bprobably\b",
    r"\bsort of\b", r"\bkind of\b", r"\bkinda\b", r"\bi'?m not sure\b", r"\bi suppose\b",
    r"\bperhaps\b", r"\bpossibly\b", r"\bor something\b", r"\bi don'?t know\b",
    r"\bhopefully\b", r"\bjust\b", r"\bactually\b", r"\bbasically\b",
]
FILLER_PATTERNS = [
    r"\bum+\b", r"\buh+\b", r"\berm+\b", r"\blike\b", r"\byou know\b", r"\bi mean\b",
    r"\bright\?", r"\bso yeah\b",
]

_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)
_FILLER_RE = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class AnswerFeatures:
    text: str
    sentences: list[str]
    words: list[str]
    content_lemmas: list[str]
    proper_nouns: list[str]
    token_vocab: frozenset[str]  # every alpha surface form + lemma, lowercased
    entities: list[tuple[str, str]]
    numeral_count: int
    sentence_lengths: list[int]
    hedge_count: int
    filler_count: int

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


def extract_features(text: str, nlp: object) -> AnswerFeatures:
    doc = nlp(text)  # type: ignore[operator]

    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sentences:
        sentences = [text.strip()] if text.strip() else []

    words = [t.text.lower() for t in doc if t.is_alpha]
    content = [
        (t.lemma_ or t.text).lower()
        for t in doc
        if t.pos_ in _CONTENT_POS and not t.is_stop and t.is_alpha
    ]
    vocab = {t.text.lower() for t in doc if t.is_alpha} | {
        (t.lemma_ or t.text).lower() for t in doc if t.is_alpha
    }
    entities = [(e.text, e.label_) for e in doc.ents]
    # PROPN catches technical proper nouns (Docker, Kubernetes, FastAPI, PyTorch)
    # that the NER model often doesn't tag as entities.
    propn = [t.text for t in doc if t.pos_ == "PROPN" and t.is_alpha and not t.is_sent_start]
    numerals = sum(1 for t in doc if t.like_num or t.pos_ == "NUM")

    sent_lengths = [len([t for t in s if t.is_alpha]) for s in doc.sents] or [len(words)]

    return AnswerFeatures(
        text=text,
        sentences=sentences,
        words=words,
        content_lemmas=content,
        proper_nouns=propn,
        token_vocab=frozenset(vocab),
        entities=entities,
        numeral_count=numerals,
        sentence_lengths=[n for n in sent_lengths if n > 0] or [len(words)],
        hedge_count=len(_HEDGE_RE.findall(text)),
        filler_count=len(_FILLER_RE.findall(text)),
    )

"""Score one strong + one weak answer for EVERY corpus question - catches
calibration misses the 5-question golden set doesn't cover."""

from __future__ import annotations

from app.core.config import get_settings
from app.services.corpus import load_corpus
from app.services.evaluator import Evaluator
from app.services.feedback.judge import NullJudge
from app.services.registry import ModelRegistry

settings = get_settings()
corpus = load_corpus(settings.corpus_path)
reg = ModelRegistry(settings, corpus)
reg.warmup()
ev = Evaluator(settings, reg, NullJudge())

# a generic-but-real strong answer template and a fluff answer
STRONG = (
    "In my last role I owned this end to end. I dug into the specifics, used Python "
    "with scikit-learn and SQL, measured the impact with clear metrics, and delivered "
    "a result that moved the number by about 20 percent over two quarters. I documented "
    "the trade-offs and walked the stakeholders through them."
)
FLUFF = (
    "This is really important to me and something I care a lot about. I always try my "
    "best and stay positive. I think being a good communicator and a team player makes "
    "a big difference here."
)

print(f"{'id':7} {'category':16} {'strong':>7} {'fluff':>7} {'gap':>6}")
for q in corpus.questions:
    s = ev.evaluate_text(q.text, STRONG).score
    f = ev.evaluate_text(q.text, FLUFF).score
    flag = "  <-- low strong" if s < 60 else ("  <-- small gap" if s - f < 15 else "")
    print(f"{q.id:7} {q.category:16} {s:7.1f} {f:7.1f} {s - f:6.1f}{flag}")

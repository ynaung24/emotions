# This makes the evaluate directory a Python package
from .evaluate import (
    evaluate_response,
    load_corpus,
    model,
    nlp
)

__all__ = [
    'evaluate_response',
    'load_corpus',
    'model',
    'nlp'
]

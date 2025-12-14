"""
tts_eval

Inference-free TTS evaluation pipeline:
- WER (Whisper)
- Speaker Similarity (WavLM-SV)
- Emotion Similarity (Emotion2Vec+)
"""

from .evaluator import (  # noqa: F401
    DATA_META,
    PROMPT_ROOT,
    load_eval_items,
    eval_worker,
)



"""
tts_eval

Inference-free TTS evaluation pipeline:
- WER (Whisper-based generic backend)
- Speaker Similarity (WavLM-SV generic backend)
- Emotion Similarity (Emotion2Vec+ generic backend)

주의:
- 패키지 임포트 시에 불필요한 의존성 에러를 피하기 위해,
  `evaluator` 의 심볼들을 여기서 바로 re-export 하지 않는다.
- 대신 필요한 경우 각 모듈에서 직접 임포트해서 사용하세요.

예:
    from tts_eval.evaluator import load_eval_items, eval_worker
"""

__all__ = []


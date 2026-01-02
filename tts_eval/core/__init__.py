"""
tts_eval.core

코어(inference-free) TTS 평가 로직 모음:
- `evaluator` : 한 샘플에 대한 WER / SS / ES 계산
- `run_eval`  : 사전 생성된 WAV 디렉터리에 대해 일괄 평가 CLI

외부에서는 다음과 같이 사용하는 것을 권장:

예)
    from tts_eval.core.evaluator import load_eval_items, eval_worker
    from tts_eval.core import run_eval  # CLI 모듈은 보통 python -m 으로 실행
"""

__all__ = []



"""
tts_eval.f5_seedtts

F5-TTS / SeedTTS 평가 재현용 서브패키지.

- `eval`       : F5-style SeedTTS 평가(WER / SIM) CLI
- `ecapa_tdnn` : ECAPA-TDNN (WavLM-large backend) 화자 임베딩 모델

이 코드는 일반적인 inference-free 평가(core) 와는 별도의,
F5-SeedTTS 실험을 재현하기 위한 전용 모듈 모음이다.
"""

__all__ = []



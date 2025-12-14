"""
WER(Word Error Rate) 계산 모듈.

- ASR 백엔드로 Whisper `large-v3` 를 사용해 합성 음성으로부터 텍스트를 복원하고,
- GT 텍스트(ref_text)와의 edit distance 기반 WER 을 계산한다.

이 모듈은 **TTS 모델과 무관하게**, 이미 생성된 WAV + ref 텍스트만 있으면
동일한 방식으로 WER을 산출하기 위한 용도로 설계되었다.
"""

import whisper
from whisper.tokenizer import get_tokenizer


########################################################
# Whisper-based WER
########################################################

# 평가 파이프라인 구동 시 한 번만 Whisper 모델을 로드한다.
# (여러 샘플을 반복 평가할 때 매번 로드하면 너무 느리기 때문)
print("[tts_eval.metrics.wer] Loading Whisper (large-v3) for WER...")
whisper_model = whisper.load_model("large-v3")
WHISPER_TOKENIZER = get_tokenizer(True)


def _normalize_text(s: str) -> str:
    """
    Whisper 텍스트 정규화 (버전별 API 차이를 고려한 호환 구현).

    - 최신 whisper: Tokenizer.normalize 가 존재할 수 있음
    - 일부 버전: whisper.normalizers.EnglishTextNormalizer 제공
    - 그 외: 소문자 + strip() 으로 최소한의 정규화만 수행
    """
    # 1) 최신 whisper: Tokenizer.normalize 가 있을 수 있음
    if hasattr(WHISPER_TOKENIZER, "normalize"):
        return WHISPER_TOKENIZER.normalize(s)  # type: ignore[attr-defined]

    # 2) 일부 버전: whisper.normalizers 제공
    try:
        from whisper.normalizers import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        return normalizer(s)
    except Exception:
        # 3) 최종 fallback: 소문자 + 양끝 공백 제거
        return s.lower().strip()


def calc_wer(wav, ref_text: str) -> float:
    """
    합성 음성(wav)과 GT 텍스트(ref_text)를 받아 WER(Word Error Rate)를 계산한다.

    Args:
        wav: 16kHz mono waveform (numpy 1D array or list-like), Whisper 입력용.
        ref_text: ground-truth 텍스트. meta.lst 의 `ref_text` 에 해당.

    Returns:
        WER (0.0 ~ 1.0 범위, 0에 가까울수록 좋음)
    """
    import editdistance

    # Whisper로 ASR 수행
    # whisper는 numpy 1D array (float32) 입력을 허용한다.
    pred_text = whisper_model.transcribe(wav, language="en")["text"]

    # ASR 결과와 GT 텍스트를 동일한 규칙으로 정규화
    hyp_n = _normalize_text(pred_text)
    ref_n = _normalize_text(ref_text)

    # 단어 단위로 분할 후 edit distance 기반 WER 계산
    hyp_words = hyp_n.split()
    ref_words = ref_n.split()
    return editdistance.eval(hyp_words, ref_words) / max(1, len(ref_words))




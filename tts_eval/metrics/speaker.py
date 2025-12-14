"""
Speaker Similarity(SS) 계산 모듈.

- 백엔드로 HuggingFace의 `microsoft/wavlm-base-plus-sv` (WavLM-SV)를 사용해
  화자 임베딩을 추출하고,
- 참조/합성 음성 간 코사인 유사도를 화자 유사도 점수로 사용한다.

이 모듈 역시 **TTS 모델과 무관하게**, 이미 생성된 WAV만 있으면
동일한 방식으로 화자 유사도를 측정하기 위한 용도로 설계되었다.
"""

import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


########################################################
# Speaker Similarity (WavLM-SV)
########################################################

# 가능한 경우 GPU를 사용하고, 그렇지 않으면 CPU를 사용한다.
device = "cuda" if torch.cuda.is_available() else "cpu"
WAVLM_SV_MODEL = "microsoft/wavlm-base-plus-sv"

print(f"[tts_eval.metrics.speaker] Loading WavLM-SV for SS: {WAVLM_SV_MODEL} on {device}...")
wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_SV_MODEL)
wavlm_sv_model = WavLMForXVector.from_pretrained(WAVLM_SV_MODEL).to(device)
wavlm_sv_model.eval()


def get_spk_emb(wav) -> torch.Tensor:
    """
    WavLM-SV 기반 화자 임베딩 추출.

    Args:
        wav: 16kHz mono waveform (numpy 1D array 등), 화자 임베딩을 추출할 음성.

    Returns:
        shape (D,) 의 1D 화자 임베딩 텐서 (L2-normalized).
    """
    # WavLM-SV feature extractor 는 (B, T) 형태의 입력을 기대하므로
    # 단일 샘플도 배치 차원으로 감싸서 전달한다.
    inputs = wavlm_feature_extractor(
        wav,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wavlm_sv_model(**inputs)
        emb = outputs.embeddings  # (batch, dim)

    # 코사인 유사도 계산 시 일관성을 위해 임베딩을 L2 정규화한다.
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.squeeze(0).cpu()


def spk_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """
    두 화자 임베딩 사이의 코사인 유사도 계산.

    Args:
        emb_a: 화자 A의 임베딩 (D,)
        emb_b: 화자 B의 임베딩 (D,)

    Returns:
        cosine similarity 스칼라 값 (−1.0 ~ 1.0, 1.0에 가까울수록 유사한 화자).
    """
    return torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()




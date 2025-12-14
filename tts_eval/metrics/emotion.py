"""
Emotion Similarity(ES) 계산 모듈.

- 백엔드로 FunASR의 `iic/emotion2vec_plus_base` (Emotion2Vec+) 를 사용해
  감정/프로소디 임베딩을 추출하고,
- 참조/합성 음성 간 코사인 유사도를 감정 유사도 점수로 사용한다.

감정 라벨이 없는 데이터셋(SeedTTS 등)에서도,
프롬프트 음성과 합성 음성 사이의 감정적 톤/프로소디 유사도를
self-supervised 임베딩 상에서 상대적으로 비교하기 위한 용도이다.
"""

import torch
import soundfile as sf
from funasr import AutoModel as FunASRAutoModel


########################################################
# Emotion Similarity (Emotion2Vec+)
########################################################

print("[tts_eval.metrics.emotion] Loading Emotion2Vec+ (emotion2vec_plus_base) for ES...")
emo_model = FunASRAutoModel(model="iic/emotion2vec_plus_base")


def get_emo_emb(wav) -> torch.Tensor:
    """
    Emotion2Vec+ 기반 감정 임베딩 추출.

    Args:
        wav: 16kHz mono waveform (numpy 1D array 등), 감정 임베딩을 추출할 음성.

    Returns:
        shape (D,) 의 1D 감정 임베딩 텐서.
        (Emotion2Vec+ 가 시계열 임베딩을 반환하는 경우 시간축 평균까지 수행)
    """
    # 현재 FunASR Emotion2Vec+ API는 파일 경로 입력을 사용하는 경우가 많아,
    # 임시 wav 파일로 한 번 저장한 뒤 해당 경로를 전달한다.
    tmp_path = "/tmp/tmp_es.wav"
    sf.write(tmp_path, wav, 16000)

    # granularity="utterance" + extract_embedding=True
    #  -> 한 발화 전체에 대한 감정 임베딩을 추출.
    result = emo_model.generate(
        input=tmp_path,
        granularity="utterance",
        extract_embedding=True,
    )

    # 모델에 따라 리스트를 반환할 수 있으므로 첫 요소를 취한다.
    if isinstance(result, list):
        if len(result) == 0:
            raise RuntimeError("Emotion2Vec returned empty result list.")
        result = result[0]

    # 반환 포맷이 버전에 따라 다를 수 있어, 우선순위대로 여러 키를 탐색한다.
    if isinstance(result, dict):
        if "feats" in result:
            emb_arr = result["feats"]
        elif "embedding" in result:
            emb_arr = result["embedding"]
        elif "embeddings" in result:
            emb_arr = result["embeddings"]
        else:
            raise KeyError(f"Unexpected Emotion2Vec result keys: {list(result.keys())}")
    else:
        raise TypeError(f"Unexpected Emotion2Vec result type: {type(result)}")

    emb = torch.tensor(emb_arr).float()
    # (T, D) 형태의 시계열 임베딩인 경우, 시간축 평균을 취해 (D,) 로 축소한다.
    if emb.ndim == 2:
        emb = emb.mean(dim=0)
    return emb


def emo_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """
    두 감정 임베딩 사이의 코사인 유사도 계산.

    Args:
        emb_a: 음성 A의 감정 임베딩 (D,)
        emb_b: 음성 B의 감정 임베딩 (D,)

    Returns:
        cosine similarity 스칼라 값 (−1.0 ~ 1.0, 1.0에 가까울수록 감정/프로소디가 유사).
    """
    return torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()




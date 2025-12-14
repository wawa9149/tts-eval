"""evaluator.py

TTS 인퍼런스 코드와 완전히 분리된 **평가 코어 모듈**.

이 모듈은:
- `meta.lst` 를 읽어서 각 샘플의 메타데이터를 정리(`load_eval_items`)
- 단일 샘플에 대해 WER / SS / ES 를 계산하는 `eval_worker` 를 제공한다.

실제 반복 평가 루프와 파일 저장은 `run_eval.py` 가 담당하며,
여기서는 "한 샘플을 어떻게 평가할 것인가?" 에만 집중한다.
"""

import os
from typing import List, Dict, Tuple

import librosa
import yaml

from .metrics.wer import calc_wer
from .metrics.speaker import get_spk_emb, spk_similarity
from .metrics.emotion import get_emo_emb, emo_similarity


########################################################
# CONFIG
########################################################

# 설정 파일 경로:
# - 기본값: tts_eval/config.yaml
# - 환경변수 EVAL_CONFIG 로 덮어쓸 수 있음
_CONFIG_PATH = os.environ.get(
    "EVAL_CONFIG",
    os.path.join(os.path.dirname(__file__), "config.yaml"),
)


def _load_config(path: str) -> Dict[str, str]:
    """
    tts_eval/config.yaml 을 읽어서 설정 딕셔너리로 반환한다.

    파일이 없으면 예외를 던져 사용자가 config 를 준비하도록 안내한다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation config file not found: {path}\n"
            "tts_eval/config.example.yaml 를 복사해서 tts_eval/config.yaml 를 만든 뒤, "
            "data_meta / prompt_root 경로를 환경에 맞게 수정하세요."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required_keys = ["data_meta", "prompt_root"]
    for k in required_keys:
        if k not in cfg:
            raise KeyError(
                f"Missing key '{k}' in {path}. "
                "tts_eval/config.example.yaml 를 참고해서 값을 채워 주세요."
            )

    return cfg


_CFG = _load_config(_CONFIG_PATH)

# 기본 설정 (어떤 데이터셋에도 사용 가능).
# - DATA_META   : 기본 meta.lst 경로 (CLI에서 덮어쓸 수 있음)
# - PROMPT_ROOT : prompt_rel 이 상대 경로일 때 기준이 되는 루트 디렉터리
DATA_META = _CFG["data_meta"]
PROMPT_ROOT = _CFG["prompt_root"]


########################################################
# Helper: meta loader
########################################################


def load_eval_items(meta_path: str) -> List[Dict]:
    """
    `meta.lst` 를 로드하여 평가용 아이템 리스트를 만든다.

    포맷:
        utt_id|ref_text|prompt_rel|synth_text

    Args:
        meta_path: meta.lst 파일 경로.

    Returns:
        dict 리스트. 각 dict 는 다음 키를 포함:
        - "utt_id": str
        - "ref_text": str
        - "prompt": 프롬프트 음성 파일의 절대 경로
        - "text": 합성에 사용된 텍스트(synth_text)
    """
    items: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            utt_id, ref_text, prompt_rel, synth_text = line.strip().split("|")
            items.append(
                {
                    "utt_id": utt_id,
                    "ref_text": ref_text,
                    "prompt": os.path.join(PROMPT_ROOT, prompt_rel),
                    "text": synth_text,
                }
            )
    return items


########################################################
# Evaluation worker
########################################################


def eval_worker(job: Tuple[str, str, str, str]):
    """
    단일 (utt_id, synth_wav, ref_text, prompt_wav) 쌍에 대해 WER / SS / ES 를 계산한다.

    Args:
        job: (utt_id, synth_wav_path, ref_text, prompt_wav_path)

    Returns:
        (utt_id, wer, ss, es) 튜플.
        - wer: Whisper 기반 WER (0.0~1.0, 낮을수록 좋음)
        - ss : WavLM-SV 기반 speaker similarity (−1.0~1.0, 높을수록 유사)
        - es : Emotion2Vec+ 기반 emotion similarity (−1.0~1.0, 높을수록 유사)
    """
    utt_id, wav_path, ref_text, prompt_path = job

    wav, _ = librosa.load(wav_path, sr=16000)
    prompt_wav, _ = librosa.load(prompt_path, sr=16000)

    # WER
    wer = calc_wer(wav, ref_text)

    # SS
    try:
        emb_pred = get_spk_emb(wav)
        emb_prompt = get_spk_emb(prompt_wav)
        ss = spk_similarity(emb_pred, emb_prompt)
    except Exception:
        ss = float("nan")

    # ES
    try:
        emo_pred = get_emo_emb(wav)
        emo_prompt = get_emo_emb(prompt_wav)
        es = emo_similarity(emo_pred, emo_prompt)
    except Exception:
        es = float("nan")

    return utt_id, wer, ss, es




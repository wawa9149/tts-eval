"""F5-style SeedTTS WER computation (EN) using faster-whisper."""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import List, Tuple

from jiwer import wer as jiwer_wer
from tqdm import tqdm


def load_asr_model_en(
    ckpt_dir: str = "",
    device: str = "cpu",
    compute_type: str = "int8",
):
    """
    SeedTTS / F5-TTS 에서 EN WER 에 사용하는 ASR 백엔드 (faster-whisper).

    Args:
        ckpt_dir: 모델 이름 또는 로컬 체크포인트 디렉터리.
                  빈 문자열("") 이면 large-v3 를 사용.
        device:  "cpu" 또는 "cuda" 등 faster-whisper 가 허용하는 device 문자열.
                 기본값은 논문 재현과 동일하게 "cpu".
        compute_type: faster-whisper 의 compute_type (기본: "int8").
    """
    from faster_whisper import WhisperModel

    model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model


def run_asr_wer(args):
    """
    F5-TTS 의 EN 전용 run_asr_wer 방식으로 WER을 계산한다.

    Args:
        args: (rank, test_set, ckpt_dir, device, compute_type)
            - rank: int, GPU index (또는 CPU 인덱스)
            - test_set: List[(gen_wav, prompt_wav, gt_text)]
            - ckpt_dir: ASR checkpoint dir (빈 문자열이면 large-v3 자동 다운로드)
    """
    rank, test_set, ckpt_dir, device, compute_type = args

    # F5 구현과 동일하게, rank 를 CUDA_VISIBLE_DEVICES 로 설정해준다.
    if device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    asr_model = load_asr_model_en(ckpt_dir=ckpt_dir, device=device, compute_type=compute_type)

    # 영어 WER에서는 ASCII punctuation 만 제거해도 충분하다.
    punctuation_all = string.punctuation
    wer_results: List[dict] = []

    for gen_wav, _prompt_wav, truth in tqdm(test_set, desc=f"ASR-WER (en) on rank {rank}"):
        segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
        hypo = ""
        for segment in segments:
            hypo = hypo + " " + segment.text

        raw_truth = truth
        raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        truth_proc = truth.lower()
        hypo_proc = hypo.lower()

        wer_val = jiwer_wer(truth_proc, hypo_proc)

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": float(wer_val),
            }
        )

    return wer_results



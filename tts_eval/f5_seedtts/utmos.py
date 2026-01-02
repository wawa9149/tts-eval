"""F5-style UTMOS evaluation, re-implemented inside tts_eval.f5_seedtts.

기능:
- 지정한 디렉터리 아래의 오디오 파일들을 순회하며
  - SpeechMOS `utmos22_strong` 모델로 UTMOS 점수를 예측하고,
  - per-utterance 점수와 평균값을 JSONL 파일로 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import torch
from tqdm import tqdm


def run_utmos(audio_dir: str, ext: str = "wav") -> Path:
    """
    주어진 디렉터리 내 오디오 파일들에 대해 UTMOS 점수를 계산하고 결과 파일 경로를 반환한다.
    """
    # 디바이스 선택: CUDA > XPU > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover - 환경 의존
        device = "xpu"
    else:
        device = "cpu"

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device)

    audio_paths = list(Path(audio_dir).rglob(f"*.{ext}"))
    utmos_score = 0.0

    utmos_result_path = Path(audio_dir) / "_utmos_results.jsonl"
    with open(utmos_result_path, "w", encoding="utf-8") as f:
        for audio_path in tqdm(audio_paths, desc="Processing UTMOS"):
            wav, sr = librosa.load(audio_path, sr=None, mono=True)
            wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)
            score = predictor(wav_tensor, sr)
            line = {"wav": str(audio_path.stem), "utmos": score.item()}
            utmos_score += score.item()
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        avg_score = utmos_score / len(audio_paths) if len(audio_paths) > 0 else 0.0
        f.write(f"\nUTMOS: {avg_score:.4f}\n")

    print(f"UTMOS: {avg_score:.4f}")
    print(f"UTMOS results saved to {utmos_result_path}")
    return utmos_result_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F5-style UTMOS evaluation inside tts_eval.f5_seedtts.")
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Root directory containing audio files for UTMOS evaluation.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="wav",
        help="Audio extension to search for (default: wav).",
    )
    return parser


def main_cli():
    args = build_argparser().parse_args()
    run_utmos(args.audio_dir, args.ext)


if __name__ == "__main__":
    main_cli()



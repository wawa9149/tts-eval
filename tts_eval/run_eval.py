"""run_eval.py

Inference-Free TTS Evaluation 엔트리 포인트 스크립트.

역할:
- `meta.lst` 와 합성 WAV 디렉터리(wav_dir)를 입력으로 받아,
- 각 샘플에 대해 evaluator.eval_worker 를 호출하여 WER / SS / ES 를 계산하고,
- per-utterance 결과(`metrics.csv`, `metrics.jsonl`) 및
  지표별 평균값(`metrics_summary.json`)을 저장한다.

이 스크립트는 **TTS 인퍼런스 코드와 완전히 독립적**이며,
어떤 TTS 모델이 생성한 wav 라도 동일한 기준으로 평가할 수 있게 한다.
"""

import os
import csv
import json
import time
import math
import argparse

from tqdm import tqdm

# 인퍼런스 코드와 분리된 평가 유틸 (Whisper / WavLM-SV / Emotion2Vec+)
from .evaluator import (  # type: ignore
    DATA_META,
    PROMPT_ROOT,  # noqa: F401  # 이 파일에서는 직접 사용하지 않지만, 설정 참고용
    load_eval_items,
    eval_worker,
)


def build_eval_jobs(meta_path: str, audio_dir: str):
    """
    기존 WAV 파일들로부터 평가 job 리스트를 구성한다.

    Args:
        meta_path: meta.lst 경로.
        audio_dir: 합성 WAV 디렉터리. 각 샘플의 파일명은 `utt_id.wav` 여야 한다.

    Returns:
        job 튜플 리스트 (utt_id, wav_path, ref_text, prompt_path)
    """
    items = load_eval_items(meta_path)
    jobs = []

    for it in items:
        utt_id = it["utt_id"]
        wav_path = os.path.join(audio_dir, f"{utt_id}.wav")

        if not os.path.exists(wav_path):
            # Skip samples that haven't been synthesized
            continue

        jobs.append((utt_id, wav_path, it["ref_text"], it["prompt"]))

    return jobs


def main():
    """
    CLI 진입점.

    예시:
        python -m tts_eval.run_eval \\
            --meta /path/to/meta.lst \\
            --wav_dir /path/to/synth_wavs \\
            --out_dir /path/to/output_dir
    """
    parser = argparse.ArgumentParser(
        description="Inference-free TTS evaluation (WER / SS / ES) for pre-generated WAV files."
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=DATA_META,
        help=f"Path to meta.lst file (default: {DATA_META})",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="Directory containing synthesized WAV files (utt_id.wav).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for metrics.csv/jsonl. Defaults to wav_dir if not set.",
    )

    args = parser.parse_args()

    meta_path = args.meta
    audio_dir = args.wav_dir
    output_dir = args.out_dir or audio_dir

    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()

    jobs = build_eval_jobs(meta_path, audio_dir)
    print(f"Found {len(jobs)} WAV files for evaluation in '{audio_dir}'.")

    csv_path = os.path.join(output_dir, "metrics.csv")
    json_path = os.path.join(output_dir, "metrics.jsonl")
    summary_path = os.path.join(output_dir, "metrics_summary.json")

    # 지표별 평균을 계산하기 위해 값들을 누적한다.
    wer_vals = []
    ss_vals = []
    es_vals = []

    eval_start = time.time()
    with open(csv_path, "w") as fc, open(json_path, "w") as fj:
        writer = csv.writer(fc)
        writer.writerow(["utt_id", "WER", "SS", "ES"])

        for job in tqdm(jobs, total=len(jobs)):
            utt_id, wer, ss, es = eval_worker(job)
            writer.writerow([utt_id, wer, ss, es])
            fj.write(json.dumps({"utt_id": utt_id, "WER": wer, "SS": ss, "ES": es}) + "\n")

            # 지표별로 값을 누적해 두었다가 나중에 평균을 계산한다.
            wer_vals.append(wer)
            ss_vals.append(ss)
            es_vals.append(es)

    eval_end = time.time()
    print(f"Evaluation time: {eval_end - eval_start:.2f} sec")

    # NaN 을 제외한 평균 계산 유틸
    def _nanmean(values):
        valid = [
            float(v)
            for v in values
            if isinstance(v, (int, float)) and not math.isnan(float(v))
        ]
        if not valid:
            return None
        return sum(valid) / len(valid)

    summary = {
        "num_samples": len(jobs),
        "WER_mean": _nanmean(wer_vals),
        "SS_mean": _nanmean(ss_vals),
        "ES_mean": _nanmean(es_vals),
    }

    with open(summary_path, "w") as fsum:
        json.dump(summary, fsum, indent=2)

    print("Summary (NaN 제외 평균):", summary)

    total_end = time.time()
    print("\n========== EVAL DONE ==========")
    print(f"Total time (eval only): {total_end - total_start:.2f} sec")
    print(f"Metrics saved:       {csv_path}")
    print(f"JSONL saved:         {json_path}")
    print(f"Summary JSON saved:  {summary_path}")


if __name__ == "__main__":
    main()



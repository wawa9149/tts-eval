"""F5-style SeedTTS evaluation script (re-implemented inside tts_eval.f5_seedtts).

이 모듈은 다음 두 메트릭용 함수들을 사용해 전체 파이프라인을 orchestration 한다.
- WER: `tts_eval.f5_seedtts.wer.run_asr_wer`
- SIM: `tts_eval.f5_seedtts.sim.run_sim`
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .wer import run_asr_wer
from .sim import run_sim


def parse_gpu_nums(gpu_nums_str: str) -> List[int]:
    """
    "8" 또는 "[0,1,2,3]" 형태의 문자열을 GPU ID 리스트로 변환.
    """
    try:
        if gpu_nums_str.startswith("[") and gpu_nums_str.endswith("]"):
            gpu_list = ast.literal_eval(gpu_nums_str)
            if isinstance(gpu_list, list):
                return list(map(int, gpu_list))
        return list(range(int(gpu_nums_str)))
    except (ValueError, SyntaxError, TypeError) as exc:  # pragma: no cover - 단순 방어 코드
        raise argparse.ArgumentTypeError(
            f"Invalid GPU specification: {gpu_nums_str}. "
            "Use a number (e.g., 8) or a list (e.g., [0,1,2,3])"
        ) from exc


def get_seed_tts_test(metalst: str, gen_wav_dir: str, gpus: Sequence[int]):
    """
    SeedTTS meta.lst와 F5-TTS 결과 디렉터리에서 평가용 (gen_wav, prompt_wav, gt_text) 리스트를 만든다.

    F5-TTS 의 f5_tts.eval.utils_eval.get_seed_tts_test 와 동일한 로직을 사용하되,
    tts_eval 안에서 독립적으로 동작하도록 재구현했다.
    """
    with open(metalst, "r", encoding="utf-8") as f:
        lines = f.readlines()

    test_set_: List[Tuple[str, str, str]] = []
    for line in tqdm(lines, desc="Loading SeedTTS meta.lst for F5-style eval"):
        parts = line.strip().split("|")
        if len(parts) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = parts
        elif len(parts) == 4:
            utt, prompt_text, prompt_wav, gt_text = parts
        else:
            # SeedTTS 포맷이 아니면 스킵
            continue

        gen_path = os.path.join(gen_wav_dir, utt + ".wav")
        if not os.path.exists(gen_path):
            # 아직 합성이 되지 않은 샘플은 스킵
            continue

        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        test_set_.append((gen_path, prompt_wav, gt_text))

    num_jobs = len(gpus)
    if num_jobs == 0:
        raise ValueError("At least one GPU (or CPU index) must be specified for F5-style eval.")

    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F5-style SeedTTS evaluation (WER / SIM) inside tts_eval.")
    parser.add_argument(
        "-e",
        "--eval_task",
        type=str,
        default="wer",
        choices=["wer", "sim"],
        help="Evaluation metric to compute.",
    )
    parser.add_argument(
        "-g",
        "--gen_wav_dir",
        type=str,
        required=True,
        help="Directory containing generated wavs (utt_id.wav) from F5-TTS.",
    )
    parser.add_argument(
        "-m",
        "--metalst",
        type=str,
        required=True,
        help="SeedTTS meta.lst path (same format as used by F5-TTS).",
    )
    parser.add_argument(
        "-n",
        "--gpu_nums",
        type=str,
        default="1",
        help="Number of GPUs to use (e.g., 8) or explicit GPU list (e.g., [0,1,2,3]). "
        "For SIM, only the first GPU will be used.",
    )
    parser.add_argument(
        "--asr_ckpt_dir",
        type=str,
        default="",
        help="Optional ASR checkpoint dir for WER. "
        "If empty, faster-whisper large-v3 will be auto-downloaded on CPU.",
    )
    parser.add_argument(
        "--asr_device",
        type=str,
        default="cpu",
        help='Device for faster-whisper ASR in WER mode (e.g., "cpu", "cuda"). '
        'Default: "cpu" (matches original SeedTTS / F5-TTS setting).',
    )
    parser.add_argument(
        "--asr_compute_type",
        type=str,
        default="int8",
        help='Compute type for faster-whisper ASR in WER mode (e.g., "int8", "int8_float16", "float16"). '
        'Default: "int8".',
    )
    parser.add_argument(
        "--wavlm_ckpt_path",
        type=str,
        required=False,
        default="",
        help="Path to WavLM-large ECAPA checkpoint (.pth) for SIM. "
        "If empty, you must set this via environment or config before running SIM.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional suffix for result file name (e.g., '_seedtts'). "
        "Result will be saved as: <gen_wav_dir>/_<eval_task>_results{suffix}.jsonl",
    )
    return parser


def main_cli():
    import multiprocessing as mp

    args = build_argparser().parse_args()

    eval_task = args.eval_task
    gen_wav_dir = args.gen_wav_dir
    metalst = args.metalst
    gpus = parse_gpu_nums(args.gpu_nums)

    full_results: List[dict] = []
    scores: List[float] = []

    if eval_task == "wer":
        test_set = get_seed_tts_test(metalst, gen_wav_dir, gpus)
        with mp.Pool(processes=len(gpus)) as pool:
            job_args = [
                (
                    rank,
                    sub_test_set,
                    args.asr_ckpt_dir,
                    args.asr_device,
                    args.asr_compute_type,
                )
                for (rank, sub_test_set) in test_set
            ]
            results_iter = pool.map(run_asr_wer, job_args)
            for r in results_iter:
                full_results.extend(r)
    elif eval_task == "sim":
        sim_gpus = [gpus[0]] if len(gpus) > 0 else [0]
        if not args.wavlm_ckpt_path:
            raise ValueError("--wavlm_ckpt_path must be provided for SIM evaluation.")
        test_set = get_seed_tts_test(metalst, gen_wav_dir, sim_gpus)
        with mp.Pool(processes=1) as pool:
            job_args = [(sim_gpus[0], sub_test_set, args.wavlm_ckpt_path) for (_rank, sub_test_set) in test_set]
            results_iter = pool.map(run_sim, job_args)
            for r in results_iter:
                full_results.extend(r)
    else:  # pragma: no cover - choices에서 이미 제한
        raise ValueError(f"Unknown eval_task: {eval_task}")

    suffix = args.output_suffix or ""
    result_path = os.path.join(gen_wav_dir, f"_{eval_task}_results{suffix}.jsonl")
    with open(result_path, "w", encoding="utf-8") as f:
        for line in full_results:
            metric_val = line[eval_task]
            scores.append(float(metric_val))
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        metric = float(round(np.mean(scores), 5)) if scores else float("nan")
        f.write(f"\n{eval_task.upper()}: {metric}\n")

    print(f"\nTotal {len(scores)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    main_cli()



"""F5-style SeedTTS speaker similarity (SIM) computation using ECAPA-TDNN + WavLM."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from .ecapa_tdnn import ECAPA_TDNN_SMALL


def run_sim(args):
    """
    F5-TTS 의 run_sim 와 동일한 방식으로 화자 유사도(SIM)를 계산.

    Args:
        args: (rank, test_set, ckpt_path)
            - rank: int, GPU index (또는 CPU 인덱스)
            - test_set: List[(gen_wav, prompt_wav, gt_text)]
            - ckpt_path: ECAPA+WavLM checkpoint (.pth) 경로
    """
    rank, test_set, ckpt_path = args
    device_str = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_path, map_location=lambda storage, _loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.to(device_str)
    model.eval()

    sim_results = []
    for gen_wav, prompt_wav, _truth in tqdm(test_set, desc=f"SIM (ECAPA+WavLM) on rank {rank}"):
        wav1, sr1 = torchaudio.load(gen_wav)
        wav2, sr2 = torchaudio.load(prompt_wav)

        if use_gpu:
            wav1 = wav1.to(device_str)
            wav2 = wav2.to(device_str)

        if sr1 != 16000:
            resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
            if use_gpu:
                resample1 = resample1.to(device_str)
            wav1 = resample1(wav1)
        if sr2 != 16000:
            resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
            if use_gpu:
                resample2 = resample2.to(device_str)
            wav2 = resample2(wav2)

        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = torch.nn.functional.cosine_similarity(emb1, emb2)[0].item()
        sim_results.append(
            {
                "wav": Path(gen_wav).stem,
                "sim": float(sim),
            }
        )

    return sim_results



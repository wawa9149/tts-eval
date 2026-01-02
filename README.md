## TTS Evaluation Pipeline

생성된 TTS 음성(WAV)을 **WER / Speaker Similarity / Emotion Similarity** 기준으로 평가하는 도구입니다.  
TTS 모델/코드베이스가 달라도, **`meta.lst` + WAV 디렉터리만 맞추면 비교 가능**합니다.
- `tts_eval.core.*` : 일반적인(inference-free) TTS 평가 파이프라인 (WER / SS / ES)
- `tts_eval.f5_seedtts.*` : F5-TTS / SeedTTS 논문 수치 재현용 F5-style 평가 스크립트

---

## Quick Start

```bash
cd tts-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m tts_eval.core.run_eval \
  --meta examples/seedtts_en/meta.lst \
  --wav_dir examples/seedtts_en/demo_wavs \
  --out_dir examples/seedtts_en/out
```

- 결과:
  - `examples/seedtts_en/out/metrics.csv` / `metrics.jsonl`
  - `examples/seedtts_en/out/metrics_summary.json`

---

## 1. 입력 준비

### 1.1 합성된 WAV 디렉터리

- **구조 예시**

```text
my_results/
  ├── utt0001.wav
  ├── utt0002.wav
  └── ...
```

- **조건**
  - 파일명(확장자 제외)은 `meta.lst` 의 `utt_id` 와 **완전히 동일**해야 합니다.

### 1.2 `meta.lst` (필수 메타 정보)

- **형식**

```text
utt_id|ref_text|prompt_rel_path|synth_text
```

- **필드 설명**
  - **utt_id**: WAV 파일 이름(확장자 제외), 예: `utt0001`
  - **ref_text**: (선택) 원본/레퍼런스 텍스트. 데이터셋이 제공하는 원문 등.
  - **prompt_rel_path**: 프롬프트(참조 화자/감정) wav 경로  
    - `PROMPT_ROOT` 기준 상대 경로이거나, 절대 경로
  - **synth_text**: **실제로 TTS 합성에 사용한 텍스트**  
    - 기본 WER 계산에서는 이 컬럼이 **GT 텍스트로 사용**됩니다.
    - 대부분의 경우 `ref_text` 와 동일하게 두면 됩니다.

- **예시**

```text
utt0001|HELLO WORLD|prompts/spk001_utt0001.wav|HELLO WORLD
```

---

## 2. `config.yaml` 설정

평가에 사용할 기본 meta 파일과 프롬프트 루트는 `tts_eval/config.yaml` 에서 지정합니다.

- 샘플 파일: `tts_eval/config.example.yaml`
- 사용 방법:

```bash
cp tts_eval/config.example.yaml tts_eval/config.yaml
```

- **예시 설정**

```yaml
data_meta: "examples/seedtts_en/meta.lst"
prompt_root: "examples/seedtts_en"
```

- 이때 `meta.lst` 에 `prompt_rel_path: "prompt-wavs/common_voice_en_10119832.wav"` 이라면,
  - 실제 경로는 `examples/seedtts_en/prompt-wavs/common_voice_en_10119832.wav` 이 됩니다.

---

## 3. 지표 정의 (간단 버전)

| Metric | Backend Model                     | **의미**                    |
| ------ | --------------------------------- | --------------------------- |
| **WER (generic)** | Whisper `large-v3`                | 텍스트 정확도 (낮을수록 좋음)   |
| **SS (generic)**  | `microsoft/wavlm-base-plus-sv`   | 화자 임베딩 유사도 (높을수록 좋음) |
| **ES (generic)**  | `iic/emotion2vec_plus_base`      | 감정/프로소디 유사도 (높을수록 좋음) |

- 위 표는 `python -m tts_eval.core.run_eval` 로 실행되는 **범용(generic) 백엔드**에 대한 설명입니다.


---

## 4. 실행 방법

### 4.1 단일 모델 평가

```bash
python -m tts_eval.core.run_eval \
  --meta /path/to/meta.lst \
  --wav_dir /path/to/synth_wavs \
  --out_dir /path/to/output_dir
```

### 4.2 F5-style SeedTTS 평가 (WER / SIM / UTMOS)

F5-TTS 레포의 동일한 알고리즘을 `tts_eval` 안에서 재구현한 엔트리포인트입니다.

#### 4.2.1 SeedTTS EN, F5-style WER

```bash
python -m tts_eval.f5_seedtts.eval \
  --eval_task wer \
  --gen_wav_dir /path/to/F5_results/seedtts_test_en/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0 \
  --metalst /path/to/data/seedtts_testset/en/meta.lst \
  --gpu_nums "[0,1,2,3]" \
  --asr_ckpt_dir ""   # 비우면 faster-whisper large-v3 자동 다운로드 (기본: CPU int8)
  # 선택 옵션 (GPU ASR 사용 예시):
  # --asr_device cuda --asr_compute_type float16
```

- 결과 파일:
  - `/path/to/F5_results/.../_wer_results.jsonl`
  - 마지막 줄에 `WER: 0.xxxxx` 형식으로 평균값이 기록됩니다. (F5 코드와 동일한 포맷)

#### 4.2.2 SeedTTS EN, F5-style SIM

```bash
python -m tts_eval.f5_seedtts.eval \
  --eval_task sim \
  --gen_wav_dir /path/to/F5_results/seedtts_test_en/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0 \
  --metalst /path/to/data/seedtts_testset/en/meta.lst \
  --gpu_nums "[0]" \
  --wavlm_ckpt_path /path/to/wavlm_large_finetune.pth
```

- 결과 파일:
  - `/path/to/F5_results/.../_sim_results.jsonl`
  - 마지막 줄에 `SIM: 0.xxxxx` 형식으로 평균값이 기록됩니다.

#### 4.2.3 SeedTTS EN, F5-style UTMOS

```bash
python -m tts_eval.f5_seedtts.utmos \
  --audio_dir /path/to/F5_results/seedtts_test_en/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0 \
  --ext wav
```

- 결과 파일:
  - `/path/to/F5_results/.../_utmos_results.jsonl`
  - 마지막 줄에 `UTMOS: 0.xxxx` 형식으로 평균값이 기록됩니다.

#### 4.2.4 F5-style 평가에 필요한 외부 체크포인트

- **SeedTTS / F5-style 평가에서 사용하는 모델들은 다음과 같습니다.**
  - English ASR Model (`--lang en`, `--asr_ckpt_dir ""` 인 경우 자동 다운로드):
    - Faster-Whisper large-v3  
      - Hugging Face: <https://huggingface.co/Systran/faster-whisper-large-v3>
  - Speaker Similarity용 WavLM 체크포인트 (SIM 필수):
    - WavLM-Large finetune 체크포인트 (`wavlm_large_finetune.pth`)  
      - F5-TTS README 에서 사용한 것과 동일한 UniSpeech WavLM 모델  
      - Google Drive: <https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view>

- **다운로드 및 경로 설정 예시**
  - 이미 F5-TTS 레포에서 위 체크포인트를 받아둔 경우:
    - 예: `/app/workspace/opensource/F5-TTS/src/f5_tts/checkpoints/UniSpeech/wavlm_large_finetune.pth`
    - 이 경로를 그대로 `--wavlm_ckpt_path` 인자로 넘기면 됩니다.
  - ASR 모델은 `--asr_ckpt_dir ""` 로 둘 경우, Faster-Whisper large-v3 가 자동 다운로드되므로 별도 설정이 필요 없습니다.

### 4.3 여러 TTS 모델 비교 (generic backend)

```bash
# 모델 A
python -m tts_eval.core.run_eval \
  --meta meta.lst \
  --wav_dir results_modelA \
  --out_dir eval_modelA

# 모델 B
python -m tts_eval.core.run_eval \
  --meta meta.lst \
  --wav_dir results_modelB \
  --out_dir eval_modelB
```

- 이후 `eval_modelA/metrics_summary.json` vs `eval_modelB/metrics_summary.json` 을 비교하면  
  동일 기준에서 WER / SS / ES 평균을 바로 비교할 수 있습니다.

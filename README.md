## Inference-Free TTS Evaluation Pipeline

이미 생성된 TTS 음성(WAV)을 **WER / Speaker Similarity / Emotion Similarity** 기준으로 평가하는 도구입니다.  
TTS 모델/코드베이스가 달라도, **`meta.lst` + WAV 디렉터리만 맞추면 공정하게 비교**할 수 있습니다.

---

## Quick Start

```bash
cd tts-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m tts_eval.run_eval \
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
  - **ref_text**: WER 계산용 정답 텍스트 (GT)
  - **prompt_rel_path**: 프롬프트(참조 화자/감정) wav 경로  
    - `PROMPT_ROOT` 기준 상대 경로이거나, 절대 경로
  - **synth_text**: TTS 입력 텍스트 (보통 `ref_text` 와 동일)

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

| Metric | Backend Model             | **의미**                    |
| ------ | ------------------------- | --------------------------- |
| **WER** | Whisper `large-v3`        | 텍스트 정확도 (낮을수록 좋음)   |
| **SS**  | `microsoft/wavlm-base-plus-sv` | 화자 임베딩 유사도 (높을수록 좋음) |
| **ES**  | `iic/emotion2vec_plus_base`    | 감정/프로소디 유사도 (높을수록 좋음) |

- 모든 모델은 **최초 실행 시 자동 다운로드**됩니다.

---

## 4. 실행 방법

### 4.1 단일 모델 평가

```bash
python -m tts_eval.run_eval \
  --meta /path/to/meta.lst \
  --wav_dir /path/to/synth_wavs \
  --out_dir /path/to/output_dir
```

### 4.2 여러 TTS 모델 비교

```bash
# 모델 A
python -m tts_eval.run_eval \
  --meta meta.lst \
  --wav_dir results_modelA \
  --out_dir eval_modelA

# 모델 B
python -m tts_eval.run_eval \
  --meta meta.lst \
  --wav_dir results_modelB \
  --out_dir eval_modelB
```

- 이후 `eval_modelA/metrics_summary.json` vs `eval_modelB/metrics_summary.json` 을 비교하면  
  동일 기준에서 WER / SS / ES 평균을 바로 비교할 수 있습니다.

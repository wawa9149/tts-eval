## Inference-Free TTS Evaluation Pipeline

이 문서는 **이미 생성된 TTS 음성(WAV)을 공통 기준으로 평가하기 위한 파이프라인**을 설명합니다.  
핵심은 다음과 같습니다:

- 인퍼런스(TTS 합성)는 **각자 모델/환경에서 수행**한다.
- 이 레포의 `tts_eval/` 코드는 **`wav 디렉터리 + meta.lst`만 있으면** WER / SS / ES를 공통 방식으로 계산한다.
- 따라서 **모델이 무엇이든, 코드베이스가 무엇이든**, 동일한 지표로 공정 비교가 가능하다.

---

### 0.1 목적 요약

- ❌ 이 레포에서 TTS 모델을 학습/인퍼런스할 필요는 없음.
- ❌ 특정 TTS 모델에 종속되지 않음.
- ✔ 이미 생성된 WAV만 있다면,
- ✔ `meta.lst`만 준비하면,
- ✔ Whisper / WavLM‑SV / Emotion2Vec+ 기반으로 **WER / Speaker Similarity / Emotion Similarity** 를 계산할 수 있음.

즉, 이 디렉터리는 **“Inference-Free TTS Evaluation Pipeline”** 을 제공합니다.

---

## 1. 전체 구조 개요

- **평가 엔트리 포인트**
  - `tts_eval/run_eval.py`
    - CLI 인터페이스 제공 (`--meta`, `--wav_dir`, `--out_dir`)
    - `meta.lst`와 WAV 디렉터리를 읽어서 WER / SS / ES 계산 후 CSV/JSONL/summary JSON 저장

- **평가 코어 로직**
  - `tts_eval/evaluator.py`
    - 메타 로더: `load_eval_items`
    - 한 샘플에 대한 평가 함수: `eval_worker(job)`

- **지표별 모듈**
  - `tts_eval/metrics/wer.py`
    - Whisper 기반 WER 계산: `calc_wer`
  - `tts_eval/metrics/speaker.py`
    - WavLM‑SV 기반 SS 계산: `get_spk_emb` + `spk_similarity`
  - `tts_eval/metrics/emotion.py`
    - Emotion2Vec+ 기반 ES 계산: `get_emo_emb` + `emo_similarity`
---

## 2. 설치

```bash
cd /home/.../tts-eval

# (옵션) 별도 가상환경 권장
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 방법 1) requirements 로 직접 설치
pip install -r requirements.txt

# 방법 2) 패키지로 설치 (개발용)
pip install -e .
```

- 루트 디렉터리의 `requirements.txt` 에 Whisper, WavLM‑SV, Emotion2Vec+ 평가에 필요한 의존성이 모두 포함되어 있습니다.
- **PyTorch / torchaudio** 는 GPU 사용 시 CUDA 버전에 맞는 wheel 을 미리 설치하는 것을 권장합니다.

Whisper / WavLM‑SV / Emotion2Vec+ 모델 가중치는 처음 평가를 실행할 때 자동으로 다운로드됩니다.

---

## 3. 평가 입력 준비 (중요)

평가를 위해 필요한 것은 **오직 두 가지**입니다.

### 3.1 Synthesized WAV 디렉터리

예:

```text
results_my_tts/
    ├── utt0001.wav
    ├── utt0002.wav
    ├── ...
```

- 파일명(확장자 제외)은 `meta.lst`의 `utt_id` 와 **완전히 동일**해야 합니다.

### 3.2 `meta.lst` (레퍼런스 정보 포함)

형식:

```text
utt_id|ref_text|prompt_rel_path|synth_text
```

- **utt_id**: WAV 파일 이름 (확장자 제외)
- **ref_text**: WER 계산용 GT 텍스트
- **prompt_rel_path**: 화자/감정 프롬프트 경로  
  - 어떤 루트 디렉터리(예: 데이터셋 루트)를 기준으로 한 상대 경로이거나,
  - 절대 경로일 수도 있습니다.
- **synth_text**: TTS 입력 텍스트 (보통 `ref_text` 와 동일)

예:

```text
common_voice_xxx-yyy|HELLO WORLD|common_voice_xxx/yyy.wav|HELLO WORLD
```

`tts_eval/evaluator.py` 의 기본 구현에서는 `prompt_rel_path` 가 `PROMPT_ROOT` 기준 상대 경로라고 가정합니다:

`PROMPT_ROOT` 는 `tts_eval/config.yaml` 에서 설정합니다 (아래 3.3 참조).

---

### 3.3 `tts_eval/config.yaml` 설정

개인 환경의 절대 경로를 코드에 직접 하드코딩하지 않기 위해,
평가에 필요한 기본 경로는 `tts_eval/config.yaml` 에서 관리합니다.

1. 샘플 설정 파일을 복사합니다.

```bash
cp tts_eval/config.example.yaml tts_eval/config.yaml
```

2. `tts_eval/config.yaml` 을 열고, 실제 환경에 맞게 수정합니다.

```yaml
data_meta: "data/seedtts_testset/en/meta.lst"   # meta.lst 경로
prompt_root: "data/seedtts_testset/en"          # prompt_rel 기준 루트 디렉터리
```

- 상대 경로를 쓰면 **레포 루트** 기준으로 해석하는 것을 권장합니다.
- LibriSpeech 등 다른 데이터셋을 평가하고 싶다면 이 값을 해당 데이터셋에 맞게 교체하면 됩니다.

---

## 4. 평가 방식 (지표 정의)

각 샘플에 대해 다음 세 가지 지표를 계산합니다.

### 4.1 WER (Word Error Rate)

- **모델**: `whisper`의 `large-v3`
- **절차**:
  1. 합성 음성(`wav`)을 Whisper로 디코딩 → `pred_text`
  2. 메타의 `ref_text` 와 함께 `calc_wer(wav, ref_text)` 호출
  3. 내부에서 Whisper 토크나이저/노멀라이저 또는 `whisper.normalizers.EnglishTextNormalizer` 를 사용해 텍스트를 정규화한 뒤, `editdistance` 기반 WER 계산

### 4.2 SS (Speaker Similarity, WavLM‑SV)

- **모델**: `microsoft/wavlm-base-plus-sv` (`WavLMForXVector`)
- **아이디어**:
  - 합성 음성(`wav`)과 프롬프트 음성(`prompt_wav`)에서 화자 임베딩을 추출하고
  - 두 임베딩의 코사인 유사도 `spk_similarity(emb_pred, emb_prompt)` 를 SS 로 사용

### 4.3 ES (Emotion Similarity, Emotion2Vec+)

- **모델**: `iic/emotion2vec_plus_base` (FunASR AutoModel)
- **아이디어**:
  - 감정 라벨이 없는 SeedTTS에서도,  
    **참조(프롬프트) 음성과 합성 음성의 Emotion2Vec+ 임베딩 사이 코사인 유사도**를 ES로 사용
  - 즉, “프롬프트의 감정/프로소디를 얼마나 비슷하게 따랐는가?” 를 보는 **self‑supervised 스타일의 상대 유사도 지표**

프롬프트(`prompt_path`)와 합성 음성(`wav_path`) 각각에서 임베딩을 뽑아 `cosine_sim` 으로 비교합니다.

---

## 5. 평가 워크플로우

### 5.1 기본 사용법

1. `meta.lst` 와 WAV 디렉터리 준비 (위 3절 참고)
2. 의존성 설치 완료
3. 아래 명령 실행:

```bash
cd /home/.../tts-eval
python -m tts_eval.run_eval \
  --meta /path/to/meta.lst \
  --wav_dir /path/to/synth_wavs \
  --out_dir /path/to/output_dir
```

4. 결과:
   - `output_dir/metrics.csv`           (utt_id, WER, SS, ES)
   - `output_dir/metrics.jsonl`         (per-utterance JSON lines)
   - `output_dir/metrics_summary.json`  (WER/SS/ES 평균값)

### 5.2 다양한 모델 비교 시 예시

예를 들어, 서로 다른 TTS 모델 A/B 에 대해:

```bash
# 모델 A 결과
python -m tts_eval.run_eval --meta meta.lst --wav_dir results_modelA --out_dir eval_modelA

# 모델 B 결과
python -m tts_eval.run_eval --meta meta.lst --wav_dir results_modelB --out_dir eval_modelB
```

- 두 디렉터리의 `metrics_summary.json` 을 비교하면,
  동일한 기준(Whisper / WavLM-SV / Emotion2Vec+)으로 계산된 평균 WER / SS / ES 를 바로 비교할 수 있습니다.

---

## 6. 해석상의 주의사항
### 6.1 WER

- Whisper‑large‑v3 기반 WER 이므로, Whisper 자체의 성능에 따라 절대 수치는 변할 수 있습니다.
- 보통 **모델 간 상대 비교**(A < B 인지 등)에 사용하는 것이 안전합니다.

### 6.2 SS (Speaker Similarity)

- WavLM‑SV 기반 화자 유사도 지표입니다.
- 도메인/녹음 조건 차이에 따라 절대값이 달라질 수 있으므로,
  절대 수치보다는 **동일 데이터/조건에서의 상대 비교**를 권장합니다.

### 6.3 ES (Emotion Similarity)

- Emotion2Vec+ 기반 감정/프로소디 유사도 지표입니다.
- SeedTTS처럼 감정 라벨이 없는 세트에서도,
  프롬프트와 합성 음성의 감정적 톤/프로소디 유사도를 비교하는 데 유용합니다.
- “감정 분류 정확도”라기보다는 **self-supervised 임베딩 상의 유사도**로 해석해야 합니다.
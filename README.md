<div align="center">

<br/>

# 🔬 GPTQ 양자화 실험 기록

### LG AI 해커톤 · EXAONE-4.0-1.2B 경량화

<img width="80%" alt="LG AI 해커톤" src="https://github.com/user-attachments/assets/9e73bcd1-05c2-4b28-a554-645f79aa28fa" />

<br/>
<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![vLLM](https://img.shields.io/badge/vLLM-0.14.1-FF6B6B?style=flat-square)
![llmcompressor](https://img.shields.io/badge/llmcompressor-GPTQ-4B32C3?style=flat-square)
![W8A8](https://img.shields.io/badge/Quantization-W8A8-00875A?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![DACON](https://img.shields.io/badge/DACON-628팀_중_71등-2F80ED?style=flat-square)

<br/>

`LG Aimers 8기`  ·  3인 팀 참가  ·  GPTQ 양자화 단독 실험

</div>

<br/>

LG AI 연구원이 주관하고 DACON에서 진행된 해커톤입니다. `EXAONE-4.0-1.2B` 모델을 vLLM 0.14.1 + L4 GPU 환경에서 양자화해서 성능(PerfNorm)과 속도(SpeedNorm)를 동시에 올리는 게 과제였습니다.

평가 공식:

```
Score = 0.5 × PerfNorm + 0.5 × SpeedNorm
```

두 지표를 동시에 올리는 게 쉬워 보이지만, 실제로는 성능을 지키면 속도가 빠지고, 속도를 올리면 성능이 깎이는 전형적인 트레이드오프입니다. 결국 핵심은 "성능을 덜 깎으면서 속도를 어떻게 올리느냐"였고, 15번 넘는 실험 끝에 W8A8 + 캘리브레이션 구간 최적화 조합으로 최고점을 냈습니다.

<br/>

---

## 📑 목차

- [🏆 결과](#-결과)
- [🧪 실험 기록](#-실험-기록)
- [💡 제일 중요했던 발견들](#-제일-중요했던-발견들)
- [🔧 최종 설정값](#-최종-설정값)
- [📂 파일 구성](#-파일-구성)
- [🚀 실행 방법](#-실행-방법)
- [🛠 기술 스택](#-기술-스택)

<br/>

---

## 🏆 결과

<div align="center">

### 628팀 중 71등 · 상위 11.3%

</div>

| 지표 | 값 |
| --- | --- |
| 최종 점수 | **0.624** |
| 기준점 대비 향상 | +8.7% (0.574 → 0.624) |
| 총 실험 횟수 | 15회 이상 |
| 탐색한 양자화 기법 | 7종 |
| 서버 에러 분석·해결 | 6건 |

<br/>

---

## 🧪 실험 기록

> 잘 된 것도, 예상 밖으로 망한 것도 그대로 기록합니다.

| 기법 | 점수 | 비고 |
| --- | --- | --- |
| W4A16 기본 | 0.574 | 출발점 |
| FP8 | 0.506 | vLLM 0.14.1에서 FP8 커널 미최적화 |
| W4A16 + actorder + dampening | 0.601 | actorder가 PerfNorm 방어의 핵심이었음 |
| QLoRA 파인튜닝 | 0.329 | 1.2B 소형 모델에서 MANTA 데이터 과적합 |
| W8A8 + actorder | 0.613 | PerfNorm 0.95로 급등, 전략 전환 결정 |
| W4A8 | 서버 에러 | L4에서 CUTLASS 커널 미지원 (H100 전용) |
| embed_tokens 양자화 포함 | 0.487 | 입력층 오차가 30개 레이어에 기하급수로 누적 |
| **W8A8 + 캘리브레이션 구간 최적화** | **0.624** | **최종 제출** |

<br/>

---

## 💡 제일 중요했던 발견들

### 캘리브레이션 데이터 구간이 이렇게 민감할 줄 몰랐다

동일한 모델, 동일한 파라미터에서 캘리브레이션 샘플 구간만 바꿨더니 점수가 달라졌습니다.

```python
# 기존
dataset["train"][:512]

# 변경 후 → +0.011점
dataset["train"][512:1024]
```

GPTQ가 Hessian 행렬 추정에 캘리브레이션 샘플을 쓰기 때문에 도메인 분포가 양자화 품질에 영향을 준다는 건 알고 있었는데, 이 정도로 민감하게 반응할 줄은 몰랐습니다. 코드 한 줄 수정으로 최고점이 바뀐 케이스라 인상적이었어요.

### 로그 없는 블랙박스 서버 디버깅

제출 서버가 완전한 블랙박스였고 실패해도 왜 실패했는지 알려주지 않았습니다. 대신 실행 시간 패턴으로 역산했습니다.

- **26초 내 종료** → 초기화 에러 (모델 로드 실패, 커널 미지원 등)
- **20분 근처 종료** → 타임아웃

이 방식으로 W4A8이 L4 GPU의 CUTLASS 커널 미지원 문제임을 파악했고, FP8도 해당 버전의 vLLM에서 최적화가 안 돼있다는 걸 확인했습니다. 에러 메시지 키워드 + 타이밍으로 총 6건의 원인을 추론해서 해결했어요.

### embed_tokens를 포함하면 안 됐던 이유

처음엔 양자화 범위를 넓힐수록 속도가 더 올라갈 거라고 생각했는데, `embed_tokens`를 포함하자마자 0.487로 폭락했습니다. 입력층에서 발생한 오차가 이후 30개 트랜스포머 레이어를 통과하면서 기하급수적으로 누적되는 게 원인이었어요. 최종적으로 `embed_tokens`와 `lm_head`는 양자화 제외로 고정했습니다.

<br/>

---

## 🔧 최종 설정값

```python
SCHEME         = "W8A8"
ACTORDER       = "dynamic"
DAMPENING_FRAC = 0.02
IGNORE         = ["embed_tokens", "lm_head"]
CALIBRATION    = dataset["train"][512:1024]
```

<br/>

---

## 📂 파일 구성

실험 번호 순서대로 정리했습니다. 숫자가 클수록 나중에 시도한 버전이고, `archive/` 폴더에는 폐기된 중간 실험들이 있습니다.

### W4A16 계열 (초기)
- **`00_sample_colab2.py` / `00_sample_local_0.57.py`** — W4A16 베이스라인. 어텐션 8비트 + MLP 4비트 혼합 정밀도도 여기서 테스트했음.
- **`03_actorder_dynamic.py`** — `actorder="dynamic"` 추가. Kaggle T4x2 환경 기반.

### W8A8 전환 (0.61x 대)
- **`13_w8a8_simple.py`** — W8A8으로 스키마 전환. 이 시점부터 점수가 확실하게 올라왔음.
- **`19_gptq_w8a8_tuned.py`** — `dampening_frac` 0.01 → 0.005로 낮춰 정확도 소폭 향상. embed_tokens 포함했다가 폭락 경험.

### 최고점 경신 시도 (0.62x 대)
- **`20_gptq_w8a8_maxpos8192.py`** — `max_position_embeddings` 8192로 축소해서 VRAM 절약 → 속도 점수 타겟팅.
- **`24_w8a8_final_tune_3.py`** — 캘리브레이션 구간 `[512:1024]`, `dampening_frac=0.03`. **최종 제출 버전**.

### 하이브리드 시도
- **`21_qlora_then_w8a8.py`** — QLoRA 파인튜닝 후 W8A8 압축. 이론상 좋아 보였는데 소형 모델에서 과적합이 너무 심해서 0.329로 크게 떨어짐. 실패 기록으로 남겨둠.

<br/>

---

## 🚀 실행 방법

Kaggle T4 x2 (32GB VRAM) 환경 기준입니다.

1. Kaggle Notebook에서 Internet Access ON, GPU Accelerator를 T4 x2로 설정
2. 원하는 스크립트를 노트북에 붙여넣고 전체 셀 실행 — 패키지 설치부터 양자화, 압축까지 스크립트 내부에서 자동 처리됨
3. DAGsHub / MLflow 연동이 내장돼 있어서 실험 추적 가능 (`DAGSHUB_USER_TOKEN` 환경변수 필요)
4. 완료 후 `/kaggle/working/`에 `.zip` 파일 자동 생성 → DACON 리더보드에 제출

<br/>

---

## 🛠 기술 스택

`vLLM 0.14.1` `llmcompressor` `GPTQ` `W8A8` `FP8` `QLoRA` `MLflow` `DAGsHub` `Python 3.11`

<br/>

---

<div align="center">

**GPTQ 양자화 실험 기록** · LG Aimers 8기 · 628팀 중 71등 (상위 11.3%)

</div>

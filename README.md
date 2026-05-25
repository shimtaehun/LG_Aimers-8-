# GPTQ Quantization Optimization (LG AI Hackathon)
- <img width="1493" height="70" alt="image" src="https://github.com/user-attachments/assets/9e73bcd1-05c2-4b28-a554-645f79aa28fa" />

이 저장소는 **LG AI Hackathon (DACON)** 대회에서 `LGAI-EXAONE/EXAONE-4.0-1.2B` 모델의 성능(PerfNorm)과 추론 속도(SpeedNorm)를 극대화하기 위해 다양한 **GPTQ 양자화(Quantization)** 최적화 기법을 실험하고 적용한 코드들을 포함하고 있습니다. 최후의 목표로 0.64 이상 달성(PerfNorm 최소화 방어 + SpeedNorm 상승)을 목표로 하고 있습니다.

## 🎯 주요 목표 및 목표 점수
- **모델**: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- **목표 점수**: 0.64 이상
- **주요 전략**: `W8A8` 양자화 기법을 통해 vLLM 환경에서 메모리를 적게 차지하고, 동시에 처리할 수 있는 시퀀스의 수를 높이는 것입니다.




## 📂 파일 구성 및 발전 과정

이 폴더 내의 스크립트들은 여러 실험 단계를 거치며 점진적으로 성능을 끌어올린 기록입니다. 핵심 실험 파일들은 다음과 같습니다:

### 1. 베이스라인 구축 및 초기 최적화 (W4A16 & Kaggle 환경)
* **`00_sample_colab2.py` / `00_sample_local_0.57.py`**
  * W4A16 양자화를 기반으로 한 초기 베이스라인 코드입니다. (약 0.57점 내외)
  * 어텐션(Attention) 레이어에는 8비트를, MLP 레이어에는 4비트 혼합 정밀도를 테스트했습니다.
* **`03_actorder_dynamic.py`**
  * Kaggle T4x2 환경 기반.
  * `actorder="dynamic"` 파라미터를 추가 적용하여 과적합을 방지하고 일반화 성능을 끌어올렸습니다.

### 2. W8A8 전환 및 최적화 점프 (0.61x 대 달성)
* **`13_w8a8_simple.py`** (0.612점 기반)
  * 가장 안정적으로 성능을 방어하는 **W8A8** 스키마로 전환.
  * `config.json`의 `max_position_embeddings`를 16384로 설정하여 VRAM 사용량을 대폭 보존하고, 속도 비율을 높였습니다.
* **`19_gptq_w8a8_tuned.py`** (0.613 성능)
  * W8A8 손실이 매우 적다는 점을 활용, `dampening_frac` 보정치를 0.01에서 0.005로 줄여 정확도를 높였습니다.
  * 기존 `embed_tokens`를 제외하고 양자화하던 것을 전체적으로 포함하여, 속도 점수(SpeedNorm)의 미세한 상승을 확보했습니다.

### 3. VRAM 효율 극대화 및 최고점 경신 시도 (0.62x 대 ~ 0.63+)
* **`20_gptq_w8a8_maxpos8192.py`**
  * 앞선 W8A8 설정에 더불어, `max_position_embeddings`를 8192로 과감하게 깎은 승부수 버전.
  * 수백 MB의 VRAM을 절약하고, 그 여유 공간으로 vLLM이 더 많은 배치를 동시 처리하게 하여 속도 점수(0.35)를 타겟팅합니다.
* **`24_w8a8_final_tune_3.py`** (최고점 0.624 모델 기반 튜닝)
  * 평가 데이터 슬라이스로 최상의 성과를 냈던 `train[512:1024]` 구간만을 사용했습니다.
  * `dampening_frac=0.03`으로 보수적으로 세팅, 오차 방어를 최대로 확보하여 안정적인 0.624 이상의 점수를 노린 최후의 마이크로 튜닝 스크립트입니다.

### 4. 하이브리드 파이프라인
* **`21_qlora_then_w8a8.py`**
  * 양자화 전에 **QLoRA**로 MANTA-1M 데이터를 학습(Fine-Tuning)시키고, 이후 원본에 어댑터를 병합(Merge)한 뒤 **W8A8** 압축을 수행합니다.
  * 양자화 오차 최소화와 함께, 모델 자체를 먼저 똑똑하게 만들어 성능 지표(`PerfNorm`)를 절대적으로 높게 유지하려는 방어 기법입니다.

## ⚙️ 핵심 양자화 파라미터 (LLMCompressor)

- **Scheme (`W8A8` > `W4A16`)**: 8비트 기반 가중치 및 활성화 정밀도 선택. EXAONE 4.0 1.2B 모델 특성상 W8A8이 가장 뛰어난 효율성을 보여주었습니다.
- **ActOrder (`dynamic`)**: 가중치 행렬을 활성화에 따라 중요도별로 양자화하여 손실을 막기 위해 필수적으로 사용되었습니다.
- **Dampening Fraction (`dampening_frac`)**: Hessian 대각 요소 안정화를 위한 값으로, 테스트 결과에 따라 0.005부터 0.03 사이의 세밀한 튜닝이 요구됩니다.
- **Max Position Embeddings 최적화**: 단순히 `config.json`을 수정하는 것만으로 VRAM의 임베딩 영역을 확 줄여, 속도 지표를 우회적으로 높이는 필살기입니다 (ex: 8192, 16384).

## 🚀 실행 방법 (Kaggle T4 x 2 환경 기준)

대부분의 코드 및 메모리 구조는 Kaggle 환경(32GB VRAM 확보) 기준으로 튜닝되어 있습니다.

1. **환경 셋업**: Kaggle Notebook에서 `Internet Access`를 ON으로 설정하고, GPU Accelerator를 `T4 x 2`로 설정합니다.
2. **실행**: 원하는 스크립트를 노트북에 붙여넣고 전체 셀을 실행합니다. (스크립트 내부에서 패키지 다운로드 및 `llmcompressor` 양자화를 모두 자동 진행합니다)
3. **로깅**: DAGsHub 및 MLflow가 연동되어 있으므로, 실행 과정을 대시보드에서 실시간 모니터링할 수 있습니다 (`DAGSHUB_USER_TOKEN` 환경변수가 필요할 수 있습니다).
4. **결과물 (제출용)**: 작업이 끝나면 `/kaggle/working/` 경로에 대시보드에 기록된 설정 이름 기반의 최적화된 `.zip` 압축 파일이 자동으로 생성됩니다. 이 파일을 다운로드하여 DACON 리더보드에 제출합니다.

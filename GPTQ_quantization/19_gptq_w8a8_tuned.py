"""
최후의 보수적 파라미터 튜닝 W8A8 (LLMCompressor)

목표: 0.613(현재 최고점) 기반의 안전한 튜닝으로 점수 상승 노리기

1. embed_tokens 양자화 포함
   - 기존 ignore=["embed_tokens", "lm_head"] → ["lm_head"]
   - 이유: 프롬프트 입력 등에서 모델 메모리 레이어 크기를 조금 더 8비트로 압축해,
           SpeedNorm(속도 점수) 미세 상승 기대

2. dampening_frac = 0.005
   - 기존: 0.01 (1%)
   - 변경: 0.005 (0.5%)
   - 이유: W8A8은 손실이 적어 양자화 오차 보정치를 살짝 낮춰 원래 가중치 형태를
           더 잘 유지하도록 하여 PerfNorm(정확도) 유지/상승 기대

이 코드는 지금까지 DACON 서버에서 유일하게 "안전하게" 최고점을 기록한
13번 파일(W8A8)과 완벽히 동일한 환경(llmcompressor, MANTA 512, 1024 seq length)에서
딱 위 두 개의 설정만 수정한 "0.613 강화판" 입니다.
"""

# =========================================================
# Kaggle 패키지 설치
# =========================================================
import subprocess
import sys

packages = [
    "llmcompressor",
    "dagshub",
    "mlflow",
    "datasets",
    "transformers>=4.40.0",
    "accelerate",
]

for pkg in packages:
    print(f"설치 중: {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("패키지 설치 완료!\n")

import os
import torch
import shutil
import json
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
# 설정
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
ZIP_NAME = "submit_w8a8_tuned"

# 캘리브레이션 (0.613 검증 설정 그대로 유지)
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

# 양자화 설정 (강화 포인트)
SCHEME = "W8A8"
ACTORDER = "dynamic"
DAMPENING_FRAC = 0.005    # 0.01 → 0.005 로 보정치 미세 조정

# 양자화 제외 레이어 (강화 포인트)
IGNORE_LAYERS = ["lm_head"]  # embed_tokens 양자화 포함하여 모델 크기 감소

# config.json 최적화 (0.613 성공 설정)
MAX_POSITION_EMBEDDINGS = 16384

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-w8a8-tuned")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# 실행
# =========================================================
print("=" * 60)
print(f"GPTQ {SCHEME} 최후의 튜닝 (0.613 강화판)")
print(f"   actorder={ACTORDER}, dampening={DAMPENING_FRAC}")
print(f"   ignore_layers={IGNORE_LAYERS}")
print(f"   samples={NUM_CALIBRATION_SAMPLES}, seq_len={MAX_SEQUENCE_LENGTH}")
print("=" * 60)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

# =========================================================
# 1. 모델 로드
# =========================================================
print("\n[1/5] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B 파라미터")

# =========================================================
# 2. 캘리브레이션 데이터 (MANTA 512, 검증된 방법 유지)
# =========================================================
print(f"\n[2/5] 캘리브레이션 데이터 {NUM_CALIBRATION_SAMPLES}개 로드 중...")
ds = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split=f"train[:{NUM_CALIBRATION_SAMPLES}]",
)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds = ds.map(preprocess)
print(f"  → {len(ds)}개 샘플 준비 완료")

# =========================================================
# 3. GPTQ W8A8 튜닝 적용 (핵심!)
# =========================================================
print(f"\n[3/5] GPTQ {SCHEME} 튜닝 양자화 시작...")
start_time = time.time()

recipe = GPTQModifier(
    scheme=SCHEME,
    targets=["Linear"],
    ignore=IGNORE_LAYERS,               # ["lm_head"] (embed_tokens까지 양자화!)
    actorder=ACTORDER,                  # "dynamic"
    dampening_frac=DAMPENING_FRAC,      # 0.005
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

quant_time = time.time() - start_time
print(f"  → 양자화 완료({quant_time:.0f}초)")

# =========================================================
# 4. 저장 + config.json 최적화
# =========================================================
print(f"\n[4/5] 모델 저장 중")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# config.json 최적화
config_path = os.path.join(OUT_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  → 모델 크기: {total_size / (1024*1024):.1f} MB (이전 0.613 때보다 줄었는지 확인 요망!)")

# =========================================================
# 5. ZIP 생성
# =========================================================
zip_name = ZIP_NAME
print(f"\n[5/5] ZIP 생성: {zip_name}.zip")
shutil.make_archive(
    base_name=f"/kaggle/working/{zip_name}",
    format="zip",
    root_dir="/kaggle/working",
    base_dir="model",
)

zip_path = f"/kaggle/working/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024*1024)

# MLflow
if USE_MLFLOW:
    with mlflow.start_run(run_name="w8a8-tuned"):
        mlflow.log_params({
            "scheme": SCHEME,
            "actorder": ACTORDER,
            "dampening_frac": DAMPENING_FRAC,
            "ignore_layers": str(IGNORE_LAYERS),
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "samples": NUM_CALIBRATION_SAMPLES,
            "seq_len": MAX_SEQUENCE_LENGTH,
        })
        mlflow.log_metrics({
            "quant_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

print("\n" + "=" * 60)
print(f"GPTQ {SCHEME} 튜닝 모델 준비 완료!")
print(f"""
0.613 강화판 설정 요약:
   스키마: {SCHEME}
   ignore_layers: {IGNORE_LAYERS} (embed_tokens 양자화 포함으로 크기 감소!)
   dampening_frac: {DAMPENING_FRAC} (0.5% 로 보정치 미세 조정!)
   캘리브레이션: MANTA {NUM_CALIBRATION_SAMPLES}개 (안전)
   max_position_embeddings: {MAX_POSITION_EMBEDDINGS}
   모델 크기: {total_size / (1024*1024):.1f} MB

{zip_name}.zip 을 DACON에 제출하세요!
""")

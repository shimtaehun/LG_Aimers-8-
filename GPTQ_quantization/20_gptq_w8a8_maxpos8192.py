"""
[최후 승부수] W8A8 + max_position_embeddings=8192

기존 0.613 코드(13번)와 완전 동일, 단 하나 다름:
max_position_embeddings: 16384 → 8192

기대 효과:
VRAM 절약: ~480MB
    → vLLM이 더 많은 요청을 동시에(배치) 처리
    → SpeedNorm 상승: 0.28 → 0.35 기대
    → Score 추정: 0.5 × 0.95 + 0.5 × 0.35 = 0.650 (0.634 목표 달성!)

안전성:
  llmcompressor + compressed-tensors (검증된 유일한 작동 포맷)
  embed_tokens, lm_head 보호 (19번 참패 교훈)
  dampening_frac = 0.01 (검증된 0.613 세팅 유지)
  한국어 벤치마크 최대 시퀀스 길이 < 8192 (안전)
"""

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
# 설정 (13번과 완전 동일, max_position_embeddings만 변경!)
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
ZIP_NAME = "submit_w8a8_maxpos8192"

# 캘리브레이션 (0.613 검증 설정 100% 유지)
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

# 양자화 설정 (0.613 검증 설정 100% 유지)
SCHEME = "W8A8"
ACTORDER = "dynamic"
DAMPENING_FRAC = 0.01
IGNORE_LAYERS = ["embed_tokens", "lm_head"]  # 19번 교훈: 절대 건드리지 말 것!

# 핵심 변경: 16384 → 8192 (VRAM 480MB 절약 → SpeedNorm 상승!)
MAX_POSITION_EMBEDDINGS = 8192   # 서버 max_gen_toks=16384의 절반, 벤치마크엔 충분

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-w8a8-maxpos8192")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# 실행
# =========================================================
print("=" * 60)
print(f"W8A8 + max_position_embeddings={MAX_POSITION_EMBEDDINGS}")
print(f"   [0.613 세팅 100% 유지 + max_pos만 절반으로!]")
print(f"   actorder={ACTORDER}, dampening={DAMPENING_FRAC}")
print(f"   ignore={IGNORE_LAYERS}")
print(f"   samples={NUM_CALIBRATION_SAMPLES}, seq_len={MAX_SEQUENCE_LENGTH}")
print("=" * 60)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)} "
            f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

# 1. 모델 로드
print("\n[1/5] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B 파라미터")

# 2. 캘리브레이션 데이터
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

# 3. GPTQ W8A8 양자화 (0.613과 100% 동일한 레시피)
print(f"\n[3/5] GPTQ {SCHEME} 양자화 시작...")
start_time = time.time()

recipe = [
    GPTQModifier(
        scheme=SCHEME,                          # "W8A8"
        targets=["Linear"],
        ignore=IGNORE_LAYERS,                   # ["embed_tokens", "lm_head"]
        actorder=ACTORDER,                      # "dynamic"
        dampening_frac=DAMPENING_FRAC,          # 0.01
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

quant_time = time.time() - start_time
print(f"  → 양자화 완료! ({quant_time:.0f}초)")

# 4. 저장
print(f"\n[4/5] 모델 저장 중...")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  → 모델 크기: {total_size / (1024*1024):.1f} MB")

# 5. config.json - max_position_embeddings를 8192로! (핵심!)
config_path = os.path.join(OUT_DIR, "config.json")
print(f"\n[5/5] config.json 최적화 중...")

with open(config_path, "r") as f:
    config = json.load(f)

original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")
vram_saved = (16384 - MAX_POSITION_EMBEDDINGS) * 30 * 8 * 64 * 4 / 1024 / 1024
print(f"  → VRAM 절약: 약 {vram_saved:.0f}MB → vLLM 배치 처리 여유 증가!")

# ZIP 생성
shutil.make_archive(
    base_name=f"/kaggle/working/{ZIP_NAME}",
    format="zip",
    root_dir="/kaggle/working",
    base_dir="model",
)

zip_path = f"/kaggle/working/{ZIP_NAME}.zip"
zip_size = os.path.getsize(zip_path) / (1024*1024)

# MLflow
if USE_MLFLOW:
    with mlflow.start_run(run_name="w8a8-maxpos8192"):
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
print(f"W8A8 + max_pos={MAX_POSITION_EMBEDDINGS} 완료!")
print(f"""
설정 (13번 0.613과의 비교):
스킴: {SCHEME} (동일)
actorder: {ACTORDER} (동일)
dampening: {DAMPENING_FRAC} (동일)
ignore: {IGNORE_LAYERS} (동일!)
samples: {NUM_CALIBRATION_SAMPLES} (동일)
max_position_embeddings: 16384 → {MAX_POSITION_EMBEDDINGS} <- 이것만 변경!

기대 점수 계산:
PerfNorm: ~0.95 (동일한 W8A8 품질 유지)
SpeedNorm: ~0.35 (VRAM {vram_saved:.0f}MB 절약 → 배치 처리 증가)
예상 Score: 0.5 × 0.95 + 0.5 × 0.35 = 0.650 → 0.634 목표 달성!

{ZIP_NAME}.zip 을 DACON에 제출하세요!
""")

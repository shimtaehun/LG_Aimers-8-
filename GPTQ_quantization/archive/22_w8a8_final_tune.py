"""
0.613 마이크로 튜닝 - 마지막 안전한 최적화

기준: 13번 W8A8 (0.613 달성) 코드에서 딱 2가지만 변경!

변경 1: MANTA 캘리브레이션 슬라이스
  기존: train[:512]   (0~511번 샘플)
  변경: train[512:1024] (512~1023번 샘플)
  이유: 다른 대화/도메인 커버리지 → 미세한 품질 향상 가능

변경 2: dampening_frac
  기존: 0.01 (1%)
  변경: 0.02 (2%)
  이유: 더 보수적인 GPTQ → 원본 가중치 더 잘 보존

나머지: 완전 동일 (embed_tokens 보호, actorder=dynamic, max_pos=16384)
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
# 설정 (13번 대비 2가지만 변경!)
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
ZIP_NAME = "submit_w8a8_finaltune"

# 변경 1: 다른 MANTA 슬라이스 (512~1023번째)
CALIBRATION_SPLIT = "train[512:1024]"  # 기존: train[:512]
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

# 양자화 (embed_tokens, lm_head 보호 - 19번 교훈 절대 유지!)
SCHEME = "W8A8"
ACTORDER = "dynamic"
IGNORE_LAYERS = ["embed_tokens", "lm_head"]

# 변경 2: dampening 증가 (0.01 → 0.02)
DAMPENING_FRAC = 0.02  # 기존: 0.01

# config.json (0.613 성공 세팅 유지!)
MAX_POSITION_EMBEDDINGS = 16384

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-w8a8-finaltune")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# 실행
# =========================================================
print("=" * 60)
print(f"W8A8 마이크로 튜닝 (13번 0.613 기반)")
print(f"   변경1: 캘리브레이션 슬라이스 = {CALIBRATION_SPLIT}")
print(f"   변경2: dampening_frac = {DAMPENING_FRAC}")
print(f"   유지: ignore={IGNORE_LAYERS}, max_pos={MAX_POSITION_EMBEDDINGS}")
print("=" * 60)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"[GPU {i}] {props.name} ({props.total_memory / 1e9:.1f}GB)")

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

# 2. 캘리브레이션 데이터 (다른 슬라이스!)
print(f"\n[2/5] 캘리브레이션 데이터 로드 중... ({CALIBRATION_SPLIT})")
ds = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split=CALIBRATION_SPLIT,
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

# 3. GPTQ W8A8 양자화
print(f"\n[3/5] GPTQ {SCHEME} 양자화 시작...")
start_time = time.time()

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=["Linear"],
        ignore=IGNORE_LAYERS,          # ["embed_tokens", "lm_head"] 보호
        actorder=ACTORDER,             # "dynamic"
        dampening_frac=DAMPENING_FRAC, # 0.02 (변경!)
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

# config.json 최적화
config_path = os.path.join(OUT_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS
with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  → 모델 크기: {total_size / (1024*1024):.1f} MB")
print(f"  → max_position_embeddings: {MAX_POSITION_EMBEDDINGS}")

# 5. ZIP 생성
print(f"\n[5/5] ZIP 생성 중...")
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
    with mlflow.start_run(run_name="w8a8-finaltune"):
        mlflow.log_params({
            "scheme": SCHEME,
            "calibration_split": CALIBRATION_SPLIT,
            "actorder": ACTORDER,
            "dampening_frac": DAMPENING_FRAC,
            "ignore_layers": str(IGNORE_LAYERS),
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "samples": NUM_CALIBRATION_SAMPLES,
        })
        mlflow.log_metrics({
            "quant_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

print("\n" + "=" * 60)
print(f"W8A8 마이크로 튜닝 완료!")
print(f"""
13번(0.613) 대비 변경 사항:
   캘리브레이션: train[:512] → {CALIBRATION_SPLIT}
   dampening_frac: 0.01 → {DAMPENING_FRAC}
   나머지 모두 동일! (에러 가능성 0%)

모델 크기: {total_size / (1024*1024):.1f} MB
ZIP 크기: {zip_size:.1f} MB

/kaggle/working/{ZIP_NAME}.zip 을 DACON에 제출하세요!
""")

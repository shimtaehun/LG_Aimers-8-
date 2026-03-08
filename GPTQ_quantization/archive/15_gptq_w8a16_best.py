"""
 GPTQ W8A16  (0.64    !)

      "    " .

 W8A16?
  - (Weight) 8bit  →
  - (Activation)  FP16  → (PerfNorm) 98%
  - () W8A8  8bit    (0.613 )
  - vLLM 0.14.1 W8A16  Marlin  GPU  (  )

 0.613    :
  - MANTA-1M  512 (   ×)
  - actorder="dynamic", dampening_frac=0.01 ( )
  - max_position_embeddings=16384 (KV Cache )
"""

# =========================================================
# Kaggle
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
    print(f"  : {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("   !\n")

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
#
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
ZIP_NAME = "submit_w8a16_best"

#  (0.613  )
NUM_CALIBRATION_SAMPLES = 768
MAX_SEQUENCE_LENGTH = 1024

# GPTQ  (  !)
SCHEME = "W8A16"             #  W8A8 → W8A16 ! (Weight 8bit, Act FP16)
ACTORDER = "dynamic"         # 0.613
DAMPENING_FRAC = 0.01        # 0.613

# config.json
MAX_POSITION_EMBEDDINGS = 16384  # 0.613

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-w8a16-best")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
#
# =========================================================
print("=" * 60)
print(f" GPTQ {SCHEME} ( 8bit,  FP16 )")
print(f"   actorder={ACTORDER}, dampening={DAMPENING_FRAC}")
print(f"   samples={NUM_CALIBRATION_SAMPLES}, seq_len={MAX_SEQUENCE_LENGTH}")
print(f"   max_position_embeddings={MAX_POSITION_EMBEDDINGS}")
print("=" * 60)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

# =========================================================
# 1.
# =========================================================
print("\n[1/5]   ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B ")

# =========================================================
# 2.   (MANTA 512)
# =========================================================
print(f"\n[2/5]   {NUM_CALIBRATION_SAMPLES}  ...")
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
print(f"  → {len(ds)}   ")

# =========================================================
# 3. GPTQ W8A16
# =========================================================
print(f"\n[3/5]  GPTQ {SCHEME}  ...")
start_time = time.time()

recipe = [
    GPTQModifier(
        scheme=SCHEME,                      # "W8A16"
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        actorder=ACTORDER,                  # "dynamic"
        dampening_frac=DAMPENING_FRAC,      # 0.01
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
print(f"  →  ! ({quant_time:.0f})")

# =========================================================
# 4.   config.json
# =========================================================
print(f"\n[4/5]   ...")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# config.json
config_path = os.path.join(OUT_DIR, "config.json")
print(f"\n  → config.json  (KV Cache )")

with open(config_path, "r") as f:
    config = json.load(f)

original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB (W8A8 )")

# =========================================================
# 5. ZIP
# =========================================================
print(f"\n[5/5] ZIP  : {ZIP_NAME}.zip")
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
    with mlflow.start_run(run_name="w8a16-best"):
        mlflow.log_params({
            "scheme": SCHEME,
            "actorder": ACTORDER,
            "dampening_frac": DAMPENING_FRAC,
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
print(f" {SCHEME} (  ) !")
print(f"""
  :
    : {SCHEME} ( 8bit,    FP16!)
    actorder: {ACTORDER}
    dampening: {DAMPENING_FRAC}
    : MANTA {NUM_CALIBRATION_SAMPLES} ( )
    max_position_embeddings: {MAX_POSITION_EMBEDDINGS}

   :
   W8A8 0.613.
   W8A16 W8A8   (PerfNorm) .
    L4 GPU Marlin    W8A8 !
     0.64   !

 {ZIP_NAME}.zip  DACON  ! !
""")

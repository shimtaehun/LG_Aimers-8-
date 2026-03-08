"""
 0.624   - +0.010

: 22 W8A8 (0.624  !)   1  !

 1: MANTA      !
  : train[512:1024] (0.624 )
  : train[1024:1536] (1024~1535 )
  :    0.011 ,           !

: 22   (damp=0.02, embed_tokens , actorder=dynamic, max_pos=16384)
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
#  (22   1 !)
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
ZIP_NAME = "submit_w8a8_slice1024"

#   1:     MANTA   (1024~1535)
CALIBRATION_SPLIT = "train[1024:1536]"  #  22: train[512:1024]
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

#  (embed_tokens, lm_head  -  !)
SCHEME = "W8A8"
ACTORDER = "dynamic"
IGNORE_LAYERS = ["embed_tokens", "lm_head"]

# 22   dampening=0.02
DAMPENING_FRAC = 0.02  # 0.624 !

# config.json (  !)
MAX_POSITION_EMBEDDINGS = 16384

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-w8a8-finetune2")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
#
# =========================================================
print("=" * 60)
print(f" W8A8   2 (0.624 )")
print(f"   1:   = {CALIBRATION_SPLIT}")
print(f"   : dampening_frac = {DAMPENING_FRAC}, ignore={IGNORE_LAYERS}")
print("=" * 60)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"[GPU {i}] {props.name} ({props.total_memory / 1e9:.1f}GB)")

# 1.
print("\n[1/5]   ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B ")

# 2.   (  1024:1536!)
print(f"\n[2/5]    ... ({CALIBRATION_SPLIT})")
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
print(f"  → {len(ds)}   ")

# 3. GPTQ W8A8
print(f"\n[3/5] GPTQ {SCHEME}  ...")
start_time = time.time()

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=["Linear"],
        ignore=IGNORE_LAYERS,          # ["embed_tokens", "lm_head"] !
        actorder=ACTORDER,             # "dynamic"
        dampening_frac=DAMPENING_FRAC, # 0.02 (0.624   !)
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

# 4.
print(f"\n[4/5]   ...")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# config.json
config_path = os.path.join(OUT_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS
with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")
print(f"  → max_position_embeddings: {MAX_POSITION_EMBEDDINGS}")

# 5. ZIP
print(f"\n[5/5] ZIP  ...")
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
    with mlflow.start_run(run_name="w8a8-finetune2"):
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
print(f" W8A8   2! !")
print(f"""
 22(0.624)   :
    : train[512:1024] → {CALIBRATION_SPLIT}
      dampening_frac: {DAMPENING_FRAC} !
      ! (  0%)

  : {total_size / (1024*1024):.1f} MB
 ZIP : {zip_size:.1f} MB

 /kaggle/working/{ZIP_NAME}.zip  DACON !
""")

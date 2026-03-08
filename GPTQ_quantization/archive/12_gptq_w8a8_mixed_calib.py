"""
 GPTQ W8A8 (INT8) +

FP8 vLLM 0.14.1    (0.454).
    **W8A8 (INT8)** .

  : W8A8  = 0.612
 +α :
  1. **W8A8 INT8 ** ( )
  2. **  ** (MANTA 256 +  256)
  3. **actorder + dampening ** ( 0.601 )

: 0.612() + α(  ) = 0.62~0.63 !
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

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
#  (Team's 0.612 Recipe + Our Optimization)
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
ZIP_NAME = "submit_w8a8_mixed"

#
NUM_CALIBRATION_SAMPLES = 512
MANTA_SAMPLES = 256
TRANS_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048  # llmcompressor  (1024 -> 2048)

# GPTQ  (W8A8 INT8)
SCHEME = "W8A8"          #  : W4A16 -> W8A8
ACTORDER = "dynamic"     #
DAMPENING_FRAC = 0.01    #

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-w8a8-mixed")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# GPU
# =========================================================
print("=" * 60)
print(f" GPTQ {SCHEME} (INT8) +  ")
print(f"   MANTA: {MANTA_SAMPLES} + : {TRANS_SAMPLES}")
print(f"   Seq Len: {MAX_SEQUENCE_LENGTH}")
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
print(f"  → : {model.num_parameters()/1e9:.2f}B")

# =========================================================
# 2.    (MANTA + )
# =========================================================
print(f"\n[2/5]    ...")

# MANTA-1M ()
print(f"  → MANTA-1M {MANTA_SAMPLES} ...")
manta_ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{MANTA_SAMPLES}]")

def preprocess_manta(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }
manta_ds = manta_ds.map(preprocess_manta).select_columns(["text"])

#   (TED Talks)
print(f"  →   {TRANS_SAMPLES} ...")
try:
    trans_ds = load_dataset("msarmi9/korean-english-multitarget-ted-talks-task", split=f"train[:{TRANS_SAMPLES*2}]")

    def preprocess_trans(example):
        en = example.get("en", example.get("english", ""))
        ko = example.get("ko", example.get("korean", ""))
        if en and ko:
            return {"text": f"Translate English to Korean.\nEnglish: {en}\nKorean: {ko}"}
        return {"text": ""}

    trans_ds = trans_ds.map(preprocess_trans).filter(lambda x: len(x["text"]) > 10).select_columns(["text"])
    if len(trans_ds) > TRANS_SAMPLES:
        trans_ds = trans_ds.select(range(TRANS_SAMPLES))

    #
    ds = concatenate_datasets([manta_ds, trans_ds]).shuffle(seed=42)
    print(f"     : {len(ds)} (MANTA + )")

except Exception as e:
    print(f"     ({e}), MANTA 512 ")
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.map(preprocess_manta).select_columns(["text"])

# =========================================================
# 3. GPTQ W8A8
# =========================================================
print(f"\n[3/5] GPTQ {SCHEME}  ...")

recipe = GPTQModifier(
    scheme=SCHEME,           # "W8A8"
    targets="Linear",
    ignore=["embed_tokens", "lm_head"],
    actorder=ACTORDER,
    dampening_frac=DAMPENING_FRAC,
)

start_time = time.time()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(ds),
)
quant_time = time.time() - start_time
print(f"  →  ! ({quant_time:.0f})")

# =========================================================
# 4.
# =========================================================
print("\n[4/5]   ...")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# config.json  (max_position_embeddings)
config_path = os.path.join(OUT_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# max_position_embeddings  (0.61364   )
MAX_POSITION_EMBEDDINGS = 16384
original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS
print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")

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
zip_size = os.path.getsize(f"/kaggle/working/{ZIP_NAME}.zip") / (1024*1024)

# MLflow
if USE_MLFLOW:
    with mlflow.start_run(run_name=f"gptq-{SCHEME.lower()}-mixed"):
        mlflow.log_params({
            "scheme": SCHEME,
            "manta": MANTA_SAMPLES,
            "trans": TRANS_SAMPLES,
            "seq_len": MAX_SEQUENCE_LENGTH
        })
        mlflow.log_metrics({"model_size_mb": total_size/1024**2, "zip_size_mb": zip_size})

print("\n" + "=" * 60)
print(f" {SCHEME} +   !")
print(f" {ZIP_NAME}.zip ({zip_size:.1f} MB)")
print("=" * 60)
print("    0.612() + α() = 0.63 !")

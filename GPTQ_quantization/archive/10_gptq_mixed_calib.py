"""
 GPTQ  +

 0.601 (04_optimized_gptq.py) GPTQ  100% !
  :

:  MANTA-1M 512 ( )
:  MANTA-1M 256 +   256 ()

→
→  ↑ ( → )
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

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
#  (0.601  100% !)
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
NUM_CALIBRATION_SAMPLES = 512   #  512
MAX_SEQUENCE_LENGTH = 1024

#
MANTA_SAMPLES = 256     #   256
TRANS_SAMPLES = 256     #   256
# : 256 + 256 = 512 ( )

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-mixed-calibration")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# GPU
# =========================================================
print("=" * 60)
print(" GPTQ +   ")
print(f"   MANTA(): {MANTA_SAMPLES} + : {TRANS_SAMPLES}")
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
# 2.
# =========================================================
print(f"\n[2/5]    ...")

# MANTA-1M ( )
print(f"  → MANTA-1M {MANTA_SAMPLES}  ...")
manta_ds = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split=f"train[:{MANTA_SAMPLES}]",
)

def preprocess_manta(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

manta_ds = manta_ds.map(preprocess_manta)
manta_ds = manta_ds.select_columns(["text"])
print(f"  → MANTA: {len(manta_ds)}  ")

#   (TED Talks ↔)
print(f"  →   {TRANS_SAMPLES}  ...")
try:
    trans_ds = load_dataset(
        "msarmi9/korean-english-multitarget-ted-talks-task",
        split=f"train[:{TRANS_SAMPLES * 2}]",  #      2
    )

    def preprocess_translation(example):
        en_text = example.get("en", example.get("english", ""))
        ko_text = example.get("ko", example.get("korean", ""))
        if en_text and ko_text:
            text = f"Translate the following English text to Korean.\n\nEnglish: {en_text}\nKorean: {ko_text}"
        else:
            text = ""
        return {"text": text}

    trans_ds = trans_ds.map(preprocess_translation)
    trans_ds = trans_ds.filter(lambda x: len(x["text"]) > 10)
    trans_ds = trans_ds.select_columns(["text"])

    #  TRANS_SAMPLES
    if len(trans_ds) > TRANS_SAMPLES:
        trans_ds = trans_ds.select(range(TRANS_SAMPLES))

    print(f"  → : {len(trans_ds)}  ")

    #
    ds = concatenate_datasets([manta_ds, trans_ds])
    ds = ds.shuffle(seed=42)

except Exception as e:
    print(f"      : {e}")
    print(f"  → MANTA-1M ")
    #   MANTA 512
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.map(preprocess_manta)
    ds = ds.select_columns(["text"])

print(f"  →   : {len(ds)}")

# =========================================================
# 3. GPTQ  (0.601   !)
# =========================================================
print("\n[3/5] GPTQ   (0.601   )...")

recipe = [
    GPTQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        actorder="dynamic",        # ← 0.601
        dampening_frac=0.01,       # ← 0.601
    )
]

start_time = time.time()

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
# 4.  + ZIP
# =========================================================
print("\n[4/5]    ZIP  ...")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")

# =========================================================
# 5. config.json
# =========================================================
MAX_POSITION_EMBEDDINGS = 32768

config_path = os.path.join(OUT_DIR, "config.json")
print(f"\n[5/5] config.json  ...")

with open(config_path, "r") as f:
    config = json.load(f)

original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")

# ZIP
zip_name = "submit_mixed_calib"
shutil.make_archive(
    base_name=f"/kaggle/working/{zip_name}",
    format="zip",
    root_dir="/kaggle/working",
    base_dir="model",
)

zip_path = f"/kaggle/working/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024*1024)
print(f"  → {zip_path} ({zip_size:.1f} MB)")

# MLflow
if USE_MLFLOW:
    with mlflow.start_run(run_name="gptq-mixed-calib"):
        mlflow.log_params({
            "manta_samples": MANTA_SAMPLES,
            "trans_samples": TRANS_SAMPLES,
            "total_samples": NUM_CALIBRATION_SAMPLES,
            "max_seq_len": MAX_SEQUENCE_LENGTH,
            "actorder": "dynamic",
            "dampening_frac": 0.01,
        })
        mlflow.log_metrics({
            "quant_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

#
print("\n" + "=" * 60)
print(" GPTQ +   !")
print(f"""
  :
   GPTQ: W4A16, actorder=dynamic, dampening=0.01 (0.601 !)
   : MANTA {MANTA_SAMPLES} +  {TRANS_SAMPLES}
    : {total_size / (1024*1024):.1f} MB

  :
   0.574: GPTQ  (MANTA 256)
   0.601: GPTQ  (MANTA 512)
   :  GPTQ  (MANTA 256 +  256)
          ↑   !

 {zip_name}.zip  DACON !
""")

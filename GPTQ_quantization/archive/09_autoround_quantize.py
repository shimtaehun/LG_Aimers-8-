"""
 AutoRound W4A16  (GPTQ  !)

AutoRound = Intel
  → (rounding)
  →  W4A16 GPTQ perplexity  (  ↑)
  → vLLM ! (compressed-tensors  auto-round )

 0.601   ,   !
"""

# =========================================================
# Kaggle
# =========================================================
import subprocess
import sys

packages = [
    "auto-round",
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

# =========================================================
#
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("autoround-quantization")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# GPU
# =========================================================
print("=" * 60)
print(" AutoRound W4A16 ")
print("=" * 60)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

# =========================================================
# 1.
# =========================================================
print("\n[1/4]   ...")
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
print(f"\n[2/4]   {NUM_CALIBRATION_SAMPLES}  ...")

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

# AutoRound
calib_texts = [example["text"] for example in ds]
print(f"  → {len(calib_texts)}   ")

# =========================================================
# 3. AutoRound
# =========================================================
print("\n[3/4] AutoRound  ...")
from auto_round import AutoRound

start_time = time.time()

autoround = AutoRound(
    model=model,
    tokenizer=tokenizer,
    bits=4,                          # W4
    group_size=128,                  # GPTQ
    sym=True,                        #
    iters=512,                       #     (!)
    seqlen=MAX_SEQUENCE_LENGTH,      #
    nsamples=NUM_CALIBRATION_SAMPLES,
    dataset=calib_texts,
)

autoround.quantize()

quant_time = time.time() - start_time
print(f"  →  ! ({quant_time:.0f})")

# =========================================================
# 4.  + ZIP
# =========================================================
print("\n[4/4]    ZIP  ...")

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

# auto_round   (vLLM )
autoround.save_quantized(
    output_dir=OUT_DIR,
    format="auto_round",  # vLLM
)
tokenizer.save_pretrained(OUT_DIR)

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")

# ZIP
zip_name = "submit_autoround"
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
    with mlflow.start_run(run_name="autoround-w4a16"):
        mlflow.log_params({
            "method": "AutoRound",
            "bits": 4,
            "group_size": 128,
            "iters": 512,
            "seqlen": MAX_SEQUENCE_LENGTH,
            "nsamples": NUM_CALIBRATION_SAMPLES,
        })
        mlflow.log_metrics({
            "quant_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

#
print("\n" + "=" * 60)
print(" AutoRound  !")
print(f"""
  :
   : AutoRound (GPTQ  rounding )
   : W4A16 (4-bit , 16-bit )
   group_size: 128
    : 512 iterations
   : {NUM_CALIBRATION_SAMPLES}
    : {total_size / (1024*1024):.1f} MB

  :
   0.574: GPTQ
   0.601: GPTQ  (actorder + dampening)
   :  AutoRound (  !)

 {zip_name}.zip  DACON !
""")

"""
 gptqmodel W8A8  (LG   !)

 : LG AI Research  GPTQ-Int8
  llmcompressor  "gptqmodel"  !
  → quant_method: "gptq" (NOT "compressed-tensors")
  → vLLM Marlin/GPTQ    →  ↑↑

  LG   100%  :
  bits=8, group_size=128, desc_act=True, sym=True
  damp_percent=0.01, true_sequential=True

+   :
  MANTA-1M 512, max_position_embeddings=16384
"""

# =========================================================
# Kaggle
# =========================================================
import subprocess
import sys

# gptqmodel  numpy   !
print(" numpy==2.2.6   (gptqmodel  )...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.2.6"])

packages = [
    "gptqmodel",
    "dagshub",
    "mlflow",
    "datasets",
    "transformers>=4.40.0",
    "accelerate",
    "optimum",
]

for pkg in packages:
    print(f"  : {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# gptqmodel   numpy
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.2.6"])

print("   !\n")

import os
import torch
import shutil
import json
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

# =========================================================
#  (LG  GPTQ-Int8 !)
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
ZIP_NAME = "submit_gptqmodel_w8"

#  (0.613  )
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

#   (LG  config.json  !)
BITS = 8                     # 8-bit INT8
GROUP_SIZE = 128             # LG
DESC_ACT = True              # actorder (desc_act=true)
SYM = True                   #
DAMP_PERCENT = 0.01          # dampening (LG  )
TRUE_SEQUENTIAL = True       #

# config.json
MAX_POSITION_EMBEDDINGS = 16384  # 0.613

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptqmodel-w8")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
#
# =========================================================
print("=" * 60)
print(f" gptqmodel W{BITS} (LG   !)")
print(f"   bits={BITS}, group_size={GROUP_SIZE}")
print(f"   desc_act={DESC_ACT}, sym={SYM}")
print(f"   damp={DAMP_PERCENT}, true_sequential={TRUE_SEQUENTIAL}")
print(f"   samples={NUM_CALIBRATION_SAMPLES}, seq_len={MAX_SEQUENCE_LENGTH}")
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

# =========================================================
# 2.
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

# gptqmodel
# gptqmodel
calibration_data = []
for example in ds:
    tokenized = tokenizer(
        example["text"],
        return_tensors="pt",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        padding=False,
    )
    calibration_data.append(tokenized)

print(f"  → {len(calibration_data)}   ")

# =========================================================
# 3. gptqmodel  (!)
# =========================================================
print(f"\n[3/5]  gptqmodel  ...")
print(f"  → LG  GPTQ-Int8  !")

start_time = time.time()

#   (LG  )
quantize_config = QuantizeConfig(
    bits=BITS,                          # 8
    group_size=GROUP_SIZE,              # 128
    desc_act=DESC_ACT,                  # True (actorder)
    sym=SYM,                            # True ()
    damp_percent=DAMP_PERCENT,          # 0.01
    true_sequential=TRUE_SEQUENTIAL,    # True
)

#   +
model = GPTQModel.load(
    MODEL_ID,
    quantize_config=quantize_config,
    trust_remote_code=True,
)

#
model.quantize(calibration_data)

quant_time = time.time() - start_time
print(f"  →  ! ({quant_time:.0f})")

# =========================================================
# 4.  (quant_method: "gptq" !)
# =========================================================
print(f"\n[4/5]   ... (GPTQ  )")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

model.save(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# config.json
config_path = os.path.join(OUT_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# quant_method  (gptq !)
quant_method = config.get("quantization_config", {}).get("quant_method", "unknown")
print(f"  → quant_method: {quant_method} {' GPTQ !' if quant_method == 'gptq' else '  '}")

# max_position_embeddings
original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")

# =========================================================
# 5. ZIP
# =========================================================
print(f"\n[5/5] ZIP : {ZIP_NAME}.zip")
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
    with mlflow.start_run(run_name="gptqmodel-w8-official"):
        mlflow.log_params({
            "library": "gptqmodel",
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "desc_act": DESC_ACT,
            "sym": SYM,
            "damp_percent": DAMP_PERCENT,
            "true_sequential": TRUE_SEQUENTIAL,
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
print(f" gptqmodel W{BITS}  !")
print(f"""
   (LG  GPTQ-Int8 ):
    : gptqmodel (NOT llmcompressor!)
    bits: {BITS}
    group_size: {GROUP_SIZE}
    desc_act: {DESC_ACT} (actorder)
    sym: {SYM} ( )
    damp_percent: {DAMP_PERCENT}
    true_sequential: {TRUE_SEQUENTIAL}
    : MANTA {NUM_CALIBRATION_SAMPLES}
    max_position_embeddings: {MAX_POSITION_EMBEDDINGS}
    quant_method: {quant_method} (→ vLLM Marlin !)

  :
    (llmcompressor): quant_method = "compressed-tensors" →
    (gptqmodel):     quant_method = "gptq"              → Marlin !

 {ZIP_NAME}.zip  DACON ! 0.64  !
""")

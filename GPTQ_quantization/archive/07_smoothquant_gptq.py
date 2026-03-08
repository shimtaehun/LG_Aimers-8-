"""
 0.62+

  2   :

 1: SmoothQuant + GPTQ ( !)
  → SmoothQuant
  →    →  ↑
  →  :  GPTQ  perplexity 0.5~2%

 2: group_size=64 (  )
  → 128
  →  :   ,

  : EXPERIMENT  A/B/C/D
"""

# =========================================================
# Kaggle   (   )
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

# SmoothQuant import (   )
try:
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    HAS_SMOOTHQUANT = True
    print(" SmoothQuantModifier  !")
except ImportError:
    try:
        from llmcompressor.modifiers import SmoothQuantModifier
        HAS_SMOOTHQUANT = True
        print(" SmoothQuantModifier  ! ( )")
    except ImportError:
        HAS_SMOOTHQUANT = False
        print(" SmoothQuantModifier  → GPTQ ")

# =========================================================
#    ( !)
# =========================================================
EXPERIMENT = "A"  # ← A, B, C, D

"""

  A    SmoothQuant + GPTQ        SQ
  B    GPTQ group_size=64        128→64
  C    SmoothQuant + group_64    SQ + g64
  D    GPTQ  ( )

"""

EXPERIMENTS = {
    # A: SmoothQuant + GPTQ ( !)
    "A": {
        "name": "smoothquant_gptq",
        "use_smoothquant": True,
        "smoothquant_alpha": 0.5,  # 0.5 =  (0.3~0.7 )
        "group_size": 128,         #
        "actorder": "dynamic",
        "dampening_frac": 0.01,
    },
    # B: group_size=64 (  )
    "B": {
        "name": "gptq_g64",
        "use_smoothquant": False,
        "smoothquant_alpha": None,
        "group_size": 64,         # 128 → 64
        "actorder": "dynamic",
        "dampening_frac": 0.01,
    },
    # C: SmoothQuant + group_size=64 ( )
    "C": {
        "name": "smoothquant_g64",
        "use_smoothquant": True,
        "smoothquant_alpha": 0.5,
        "group_size": 64,
        "actorder": "dynamic",
        "dampening_frac": 0.01,
    },
    # D:  ( 0.60148  )
    "D": {
        "name": "baseline_optimized",
        "use_smoothquant": False,
        "smoothquant_alpha": None,
        "group_size": 128,
        "actorder": "dynamic",
        "dampening_frac": 0.01,
    },
}

cfg = EXPERIMENTS[EXPERIMENT]

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
    mlflow.set_experiment("final-optimization")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
#
# =========================================================
print("=" * 60)
print(f"  {EXPERIMENT}: {cfg['name']}")
print(f"   SmoothQuant: {cfg['use_smoothquant']} (alpha={cfg['smoothquant_alpha']})")
print(f"   group_size: {cfg['group_size']}")
print(f"   actorder: {cfg['actorder']}")
print("=" * 60)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

# 1.
print("\n[1/5]   ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → : {model.num_parameters()/1e9:.2f}B")

# 2.
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

# 3.
print(f"\n[3/5]    ...")

recipe = []

# SmoothQuant  ( A, C)
if cfg["use_smoothquant"] and HAS_SMOOTHQUANT:
    sq_modifier = SmoothQuantModifier(
        smoothing_strength=cfg["smoothquant_alpha"],
    )
    recipe.append(sq_modifier)
    print(f"   SmoothQuant  (alpha={cfg['smoothquant_alpha']})")
elif cfg["use_smoothquant"] and not HAS_SMOOTHQUANT:
    print(f"   SmoothQuant   → GPTQ ")

# GPTQ
gptq_scheme = {
    "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": cfg["group_size"],
    }
}

gptq_modifier = GPTQModifier(
    scheme="W4A16",
    targets=["Linear"],
    ignore=["embed_tokens", "lm_head"],
    actorder=cfg["actorder"],
    dampening_frac=cfg["dampening_frac"],
)

# group_size 128    scheme
if cfg["group_size"] != 128:
    gptq_modifier = GPTQModifier(
        scheme=gptq_scheme,
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        actorder=cfg["actorder"],
        dampening_frac=cfg["dampening_frac"],
    )
    print(f"   GPTQ (group_size={cfg['group_size']}, actorder={cfg['actorder']})")
else:
    print(f"   GPTQ (W4A16, actorder={cfg['actorder']})")

recipe.append(gptq_modifier)

# 4.
print(f"\n[4/5]  ! (: {len(recipe)} )")
start_time = time.time()

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

quant_time = time.time() - start_time
print(f"  → : {quant_time:.1f}")

# 5.  + ZIP
print(f"\n[5/5]    ZIP  ...")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")

# ZIP
zip_name = f"submit_exp{EXPERIMENT}_{cfg['name']}"
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
    with mlflow.start_run(run_name=f"final-{EXPERIMENT}-{cfg['name']}"):
        mlflow.log_params({
            "experiment": EXPERIMENT,
            "name": cfg["name"],
            "smoothquant": cfg["use_smoothquant"],
            "sq_alpha": str(cfg["smoothquant_alpha"]),
            "group_size": cfg["group_size"],
            "actorder": cfg["actorder"],
            "dampening_frac": cfg["dampening_frac"],
            "samples": NUM_CALIBRATION_SAMPLES,
            "seq_len": MAX_SEQUENCE_LENGTH,
        })
        mlflow.log_metrics({
            "quant_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

#
print("\n" + "=" * 60)
print(f"  {EXPERIMENT} ({cfg['name']}) !")
print(f"""
  :
   SmoothQuant: {cfg['use_smoothquant']} (alpha={cfg['smoothquant_alpha']})
   group_size: {cfg['group_size']}
   actorder: {cfg['actorder']}
   dampening: {cfg['dampening_frac']}
    : {total_size / (1024*1024):.1f} MB

 :
   0.574 :  GPTQ (samples=256, seq_len=512, g128)
   0.601 :  GPTQ (samples=512, seq_len=1024, actorder, dampening)
    : {'SmoothQuant + ' if cfg['use_smoothquant'] else ''}GPTQ (g{cfg['group_size']})

 {zip_name}.zip  DACON !

   : A → C → B
""")

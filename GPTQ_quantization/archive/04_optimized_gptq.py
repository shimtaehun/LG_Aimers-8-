"""
GPTQ   3 - 0.574 → 0.60 !

0.574  :
  - samples=256, seq_len=512, actorder=, dampening=
  - →  ,   !

  3 :
  A)   (samples↑, seq_len↑)
  B) A + actorder="dynamic" + dampening_frac=0.01
  C) A + B + ignore=["lm_head"] (embed_tokens  )

:
  Kaggle EXPERIMENT  "A", "B", "C"  .
     .
"""

import os
import torch
import shutil
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
#   : "A", "B", "C"
# =========================================================
EXPERIMENT = "B"  # ←   !

# =========================================================
#
# =========================================================
EXPERIMENTS = {
    # A:   ( )
    "A": {
        "name": "calib_boost",
        "num_calibration_samples": 512,    # 256 → 512 (T4 )
        "max_sequence_length": 1024,       # 512 → 1024 (T4 )
        "scheme": "W4A16",
        "targets": ["Linear"],
        "ignore": ["embed_tokens", "lm_head"],  #
        "gptq_kwargs": {},  #
    },
    # B:  + actorder + dampening (!)
    "B": {
        "name": "actorder_dynamic",
        "num_calibration_samples": 512,    # T4
        "max_sequence_length": 1024,       # T4
        "scheme": "W4A16",
        "targets": ["Linear"],
        "ignore": ["embed_tokens", "lm_head"],
        "gptq_kwargs": {
            "actorder": "dynamic",      #
            "dampening_frac": 0.01,     #
        },
    },
    # C: B + ignore  (embed_tokens )
    "C": {
        "name": "aggressive",
        "num_calibration_samples": 512,    # T4
        "max_sequence_length": 1024,       # T4
        "scheme": "W4A16",
        "targets": ["Linear"],
        "ignore": ["lm_head"],  #  embed_tokens  →
        "gptq_kwargs": {
            "actorder": "dynamic",
            "dampening_frac": 0.01,
        },
    },
}

cfg = EXPERIMENTS[EXPERIMENT]

# =========================================================
#
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

# DagsHub MLflow ()
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-tuning")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
#
# =========================================================
print("=" * 60)
print(f"  {EXPERIMENT}: {cfg['name']}")
print(f"   samples={cfg['num_calibration_samples']}, "
      f"seq_len={cfg['max_sequence_length']}")
print(f"   ignore={cfg['ignore']}")
print(f"   gptq_kwargs={cfg['gptq_kwargs']}")
print("=" * 60)

# GPU
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
print(f"  → {model.num_parameters()/1e9:.2f}B ")

# 2.
print(f"\n[2/5]   {cfg['num_calibration_samples']}  ...")
ds = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split=f"train[:{cfg['num_calibration_samples']}]",
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

# 3. GPTQ
print(f"\n[3/5] GPTQ  ...")
start_time = time.time()

recipe = [
    GPTQModifier(
        scheme=cfg["scheme"],
        targets=cfg["targets"],
        ignore=cfg["ignore"],
        **cfg["gptq_kwargs"],  # actorder, dampening_frac
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=cfg["max_sequence_length"],
    num_calibration_samples=cfg["num_calibration_samples"],
)

quant_time = time.time() - start_time
print(f"  → : {quant_time:.1f}")

# 4.
print(f"\n[4/6]   ... → {OUT_DIR}")
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

#
total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")

# =========================================================
# config.json
# =========================================================
# vLLM max_model_len   config.json max_position_embeddings .
# EXAONE  65536 →  KV cache   → batch↑ → ↑
import json

MAX_POSITION_EMBEDDINGS = 32768    # 65536 → 32768 (,  )
# MAX_POSITION_EMBEDDINGS = 16384  #   (max_gen_toks )

config_path = os.path.join(OUT_DIR, "config.json")
print(f"\n[5/6] config.json  ...")

with open(config_path, "r") as f:
    config = json.load(f)

original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")
print(f"  → KV cache  {(1 - MAX_POSITION_EMBEDDINGS/65536)*100:.0f}%  !")

# 6. ZIP
print(f"\n[6/6] ZIP  ...")
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
    with mlflow.start_run(run_name=f"exp{EXPERIMENT}-{cfg['name']}"):
        mlflow.log_params({
            "experiment": EXPERIMENT,
            "samples": cfg["num_calibration_samples"],
            "seq_len": cfg["max_sequence_length"],
            "scheme": cfg["scheme"],
            "ignore": str(cfg["ignore"]),
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            **{f"gptq_{k}": str(v) for k, v in cfg["gptq_kwargs"].items()},
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
   (0.574 ):
   samples: 256 → {cfg['num_calibration_samples']}
   seq_len: 512 → {cfg['max_sequence_length']}
   actorder:  → {cfg['gptq_kwargs'].get('actorder', '')}
   dampening:  → {cfg['gptq_kwargs'].get('dampening_frac', '')}
   ignore: embed+lm_head → {cfg['ignore']}
    max_position_embeddings: 65536 → {MAX_POSITION_EMBEDDINGS}

 {zip_name}.zip  DACON !
""")

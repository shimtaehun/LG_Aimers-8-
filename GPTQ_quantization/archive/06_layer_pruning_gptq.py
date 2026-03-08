"""
 0.64   :   + GPTQ + config.json

:
  EXAONE-4.0-1.2B = 36
  →  4~8   (~   )
  →   11~22%  +
  →  95~99%  (/  )
  → GPTQ
  → config.json max_position_embeddings

      +    =  !

 (Kaggle):
  LAYERS_TO_REMOVE
  4  ,   6~8

: trust_remote_code=True  (EXAONE  )
"""

import os
import torch
import shutil
import json
import time
import copy
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
#
# =========================================================
LAYERS_TO_REMOVE = 4       #    (4, 6, 8 )
MAX_POSITION_EMBEDDINGS = 16384  # config.json

#
EXPERIMENT_NAME = f"prune{LAYERS_TO_REMOVE}_gptq_pos{MAX_POSITION_EMBEDDINGS}"

# =========================================================
#
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"

# GPTQ
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]

# DagsHub MLflow ()
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("layer-pruning")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# 1.
# =========================================================
print("=" * 60)
print(f" : {EXPERIMENT_NAME}")
print(f"    : {LAYERS_TO_REMOVE}")
print(f"   max_position_embeddings: {MAX_POSITION_EMBEDDINGS}")
print("=" * 60)

# GPU
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

print("\n[1/6]   ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

#
total_layers = model.config.num_hidden_layers
print(f"  →   : {total_layers}")
print(f"  →  : {model.num_parameters()/1e9:.2f}B")

# =========================================================
# 2.
# =========================================================
print(f"\n[2/6]  : {total_layers} → {total_layers - LAYERS_TO_REMOVE}")

#
#  :  ( )
#  2 +  2
start_remove = total_layers - LAYERS_TO_REMOVE - 2  #  2
end_remove = total_layers - 2

layers_to_keep = []
for i in range(total_layers):
    if i < start_remove or i >= end_remove:
        layers_to_keep.append(i)

print(f"  →  : {list(range(start_remove, end_remove))}")
print(f"  →  : {layers_to_keep}")

#
# EXAONE
try:
    #  1: model.model.layers  ( HF )
    old_layers = model.model.layers
    new_layers = torch.nn.ModuleList([old_layers[i] for i in layers_to_keep])
    model.model.layers = new_layers
except AttributeError:
    try:
        #  2: model.transformer.h  (GPT )
        old_layers = model.transformer.h
        new_layers = torch.nn.ModuleList([old_layers[i] for i in layers_to_keep])
        model.transformer.h = new_layers
    except AttributeError:
        print("     !")
        print(f"   : {type(model)}")
        print(f"   : {[name for name, _ in model.named_children()]}")
        raise

# config
model.config.num_hidden_layers = total_layers - LAYERS_TO_REMOVE

print(f"  →   : {model.config.num_hidden_layers}")
print(f"  →   : {model.num_parameters()/1e9:.2f}B")

# =========================================================
# 3.
# =========================================================
print(f"\n[3/6]   {NUM_CALIBRATION_SAMPLES}  ...")

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
# 4. GPTQ
# =========================================================
print(f"\n[4/6] GPTQ  ...")
start_time = time.time()

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        actorder="dynamic",
        dampening_frac=0.01,
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
print(f"  → : {quant_time:.1f}")

# =========================================================
# 5.  + config.json
# =========================================================
print(f"\n[5/6]    config.json  ...")

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# config.json
config_path = os.path.join(OUT_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

original_max_pos = config.get("max_position_embeddings", "N/A")
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS

with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"  → max_position_embeddings: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}")
print(f"  → num_hidden_layers: {total_layers} → {total_layers - LAYERS_TO_REMOVE}")

#
total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")

# =========================================================
# 6. ZIP
# =========================================================
print(f"\n[6/6] ZIP  ...")
zip_name = f"submit_{EXPERIMENT_NAME}"
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
    with mlflow.start_run(run_name=EXPERIMENT_NAME):
        mlflow.log_params({
            "layers_removed": LAYERS_TO_REMOVE,
            "original_layers": total_layers,
            "remaining_layers": total_layers - LAYERS_TO_REMOVE,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
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
print(f" {EXPERIMENT_NAME} !")
print(f"""
  :
   : {total_layers} → {total_layers - LAYERS_TO_REMOVE} ({LAYERS_TO_REMOVE} )
   : W4A16 + actorder=dynamic
   max_pos: {original_max_pos} → {MAX_POSITION_EMBEDDINGS}
    : {total_size / (1024*1024):.1f} MB
   ZIP : {zip_size:.1f} MB

 {zip_name}.zip  DACON !

 :   LAYERS_TO_REMOVE = 6  8 !
""")

"""
GPTQ    - Kaggle Notebook  (T4 x 2)

:
1. Kaggle Notebook  (Accelerator: GPU T4 x 2 )
2. Internet Access: ON
3.

:
- 00_sample_colab3.py  Kaggle
- (32GB)
"""

# =========================================================
# 0.   (Kaggle  !)
# =========================================================
# !pip install -q llmcompressor transformers datasets accelerate dagshub mlflow

import os
os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
import torch
import shutil
import time
from pathlib import Path
import mlflow
import dagshub

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
# DagsHub + MLflow
# =========================================================
dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
mlflow.set_experiment("sth-scheme")

# =========================================================
# 1.   (Kaggle)
# =========================================================
# HuggingFace
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

#   (Kaggle : /kaggle/working)
OUT_DIR = "/kaggle/working/model"

# =========================================================
# 2.
# =========================================================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

#  Kaggle T4 x 2 (32GB VRAM)
# Colab  ,  VRAM
NUM_CALIBRATION_SAMPLES = 256   # (Colab 512 -> Kaggle 1024)
MAX_SEQUENCE_LENGTH = 512      # (Colab 1024 -> Kaggle 2048)

# =========================================================
# 3.   ()
# =========================================================
SCHEME = "W8A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
ACTORDER = "dynamic"           #  ,
DAMPENING_FRAC = 0.01

# =========================================================
# 4. GPU
# =========================================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[INFO] GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"       VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] GPU   . CPU  ( ).")

# =========================================================
# MLflow
# =========================================================
with mlflow.start_run(run_name=f"scheme-{SCHEME}"):

    # (params)
    mlflow.log_params({
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "calibration_samples": NUM_CALIBRATION_SAMPLES,
        "max_seq_length": MAX_SEQUENCE_LENGTH,
        "quantization_scheme": SCHEME,
        "targets": str(TARGETS),
        "ignore": str(IGNORE),
        "actorder": ACTORDER,
        "dampening_frac": DAMPENING_FRAC,
    })

# =========================================================
# 5.
# =========================================================
    print("\n" + "=" * 60)
    print(f"[INFO]   ... ({MODEL_ID})")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # T4 bfloat16  float16
        device_map="auto",          # Multi-GPU
        trust_remote_code=True,
    )

    print(f"[INFO]   !")
    print(f"       : {model.num_parameters() / 1e9:.2f}B")

# =========================================================
# 6.   &
# =========================================================
    print("\n" + "=" * 60)
    print(f"[INFO]    ...")
    print(f"       : {DATASET_ID}")
    print(f"        : {NUM_CALIBRATION_SAMPLES}")
    print("=" * 60)

    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    )

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False
            )
        }

    ds = ds.map(preprocess)
    print(f"[INFO]    ({len(ds)} )")

# =========================================================
# 7. GPTQ
# =========================================================
    print("\n" + "=" * 60)
    print("[INFO] GPTQ   ( 10~20 )")
    print(f"       Scheme: {SCHEME}")
    print(f"       ActOrder: {ACTORDER}")
    print(f"       Max Seq Length: {MAX_SEQUENCE_LENGTH}")
    print("=" * 60)

    recipe = [
        GPTQModifier(
            scheme=SCHEME,
            targets=TARGETS,
            ignore=IGNORE,
            actorder=ACTORDER,
            dampening_frac=DAMPENING_FRAC,
            # sequential_targets=["Exaone4DecoderLayer"], #  T4   (OOM )
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

    quantization_time = time.time() - start_time
    print(f"[INFO] GPTQ  ! ( : {quantization_time:.1f})")

    #
    mlflow.log_metric("quantization_time_sec", quantization_time)

# =========================================================
# 8.   (Kaggle )
# =========================================================
    print(f"\n[INFO]   ... → {OUT_DIR}")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    #
    print("[INFO]  :")
    for f in os.listdir(OUT_DIR):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / (1024 * 1024)
        print(f"       - {f} ({size:.1f} MB)")

# =========================================================
# 9. ZIP  (Kaggle Output)
# =========================================================
    zip_name = "kaggle_optimized_submit"
    print(f"\n[INFO] {zip_name}.zip  ...")

    # Kaggle Working
    shutil.make_archive(
        base_name=f"/kaggle/working/{zip_name}",
        format="zip",
        root_dir="/kaggle/working",
        base_dir="model",
    )

    zip_path = f"/kaggle/working/{zip_name}.zip"
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[INFO]  : {zip_path} ({zip_size:.1f} MB)")

    # ZIP
    mlflow.log_metric("model_zip_size_MB", zip_size)

    # GPU
    if torch.cuda.is_available():
        mlflow.log_param("gpu_count", torch.cuda.device_count())
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        mlflow.log_metric("gpu_vram_GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    print("[INFO] MLflow  !")

    # =========================================================
    # !
    # =========================================================
    print("\n" + "=" * 60)
    print("  !")
    print("=" * 60)
    print(f"""
  :
   • Model: {MODEL_ID}
   • Scheme: {SCHEME}
   • ActOrder: {ACTORDER}
   • Samples: {NUM_CALIBRATION_SAMPLES}
   • Max Length: {MAX_SEQUENCE_LENGTH}
   •  : {quantization_time:.1f}

 :
   • : {OUT_DIR}/
   • ZIP: {zip_path} ({zip_size:.1f} MB)

 Kaggle  'Data'  Output !
 DagsHub   : https://dagshub.com/sthun0211/LGaimers.mlflow
""")

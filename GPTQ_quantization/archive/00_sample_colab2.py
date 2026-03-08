"""
GPTQ    - Google Colab

:
1. Colab GPU   (T4  )
2.

: Colab      OOM
     → num_calibration_samples max_seq_length
"""


import os
import torch
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"


OUT_DIR = "/content/model"


DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

#  Colab
# - Colab Pro: 512 , 1024
# - Colab Free: 256 , 512
NUM_CALIBRATION_SAMPLES = 512  #    128
MAX_SEQUENCE_LENGTH = 1024      #    256


SCHEME = "W4A16"  
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
ACTORDER = "static"          

#
SYMMETRIC = False              
SEQUENTIAL_TARGETS = ["Exaone4DecoderLayer"]  
OFFLOAD_HESSIANS = False       

#     ( 8 )
CONFIG_GROUPS = {
    "attention": {
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "scheme": "W8A16",   
    },
    "mlp": {
        "targets": ["gate_proj", "up_proj", "down_proj"],
        "scheme": "W4A16",     
    }
}


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] GPU   . CPU  ( ).")


print("\n" + "=" * 60)
print(f"[INFO]   ... ({MODEL_ID})")
print("       (    2.5GB )")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"[INFO]   !")
print(f"       : {model.num_parameters() / 1e9:.2f}B")


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
print("[INFO] GPTQ   ( 5~10 )")
print(f"       Scheme: {SCHEME}")
print(f"       ActOrder: {ACTORDER}")
print(f"       Max Seq Length: {MAX_SEQUENCE_LENGTH}")
print("=" * 60)

recipe = [
    GPTQModifier(
        config_groups=CONFIG_GROUPS,
        ignore=IGNORE,
        actorder=ACTORDER,
        dampening_frac=DAMPENING_FRAC,
        symmetric=SYMMETRIC,
        block_size=BLOCK_SIZE,
        sequential_targets=SEQUENTIAL_TARGETS,
        offload_hessians=OFFLOAD_HESSIANS,
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] GPTQ  !")


print(f"\n[INFO]   ... → {OUT_DIR}")

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

#
print("[INFO]  :")
for f in os.listdir(OUT_DIR):
    size = os.path.getsize(os.path.join(OUT_DIR, f)) / (1024 * 1024)
    print(f"       - {f} ({size:.1f} MB)")


zip_name = "optimized_submit"
print(f"\n[INFO] {zip_name}.zip  ...")

shutil.make_archive(
    base_name=f"/content/{zip_name}",
    format="zip",
    root_dir="/content",
    base_dir="model",
)

zip_path = f"/content/{zip_name}.zip"
zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print(f"[INFO]  : {zip_path} ({zip_size:.1f} MB)")

# Colab
try:
    from google.colab import files
    print("\n[INFO]   ...")
    files.download(zip_path)
except ImportError:
    print(f"\n[INFO] Colab  .  : {zip_path}")


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

 :
   • : {OUT_DIR}/
   • ZIP: {zip_path}

  ZIP   !
""")

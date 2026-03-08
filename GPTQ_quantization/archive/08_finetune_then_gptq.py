"""
  : FP16  → GPTQ  ( !)

:  QLoRA   "  "
      : FP16  → GPTQ  →

:
  1. FP16 EXAONE
  2.  +   LoRA
  3. LoRA   (FP16 )
  4.   GPTQ
  5. ZIP  → !

Kaggle T4 x2
: ~2~3
"""

# =========================================================
# Kaggle
# =========================================================
import subprocess
import sys

packages = [
    "llmcompressor",
    "peft",
    "trl",
    "bitsandbytes",
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
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
#
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "/kaggle/working/model"
LORA_OUTPUT = "/kaggle/working/lora_adapter"
MERGED_OUTPUT = "/kaggle/working/merged_model"

#   (  )
LORA_R = 32               # 32: 1.2B    - rank
LORA_ALPHA = 64            # alpha = 2 * r ( )
LORA_DROPOUT = 0.05
NUM_TRAIN_EPOCHS = 2       # 2 epoch ( ,  )
LEARNING_RATE = 2e-4       # LoRA
MAX_SEQ_LENGTH = 512       # T4
PER_DEVICE_BATCH = 2
GRADIENT_ACCUMULATION = 4  #   = 2 * 4 = 8

# GPTQ
NUM_CALIBRATION_SAMPLES = 512
GPTQ_MAX_SEQ_LENGTH = 1024

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("finetune-then-gptq")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# GPU
# =========================================================
print("=" * 60)
print("  → GPTQ  ")
print("=" * 60)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

# =========================================================
# 1:  &
# =========================================================
print("\n[1/6]   ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → : {model.num_parameters()/1e9:.2f}B")

# =========================================================
# 2:    ( +  )
# =========================================================
print("\n[2/6]    ...")

# MANTA-1M (EXAONE   )
print("  → MANTA-1M  ...")
manta_ds = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split="train[:2000]",  # 2000  ( )
)

def preprocess_manta(example):
    text = tokenizer.apply_chat_template(
        example["conversations"],
        add_generation_prompt=True,
        tokenize=False,
    )
    return {"text": text}

manta_ds = manta_ds.map(preprocess_manta)

#   (TED Talks ↔)
print("  →    ...")
try:
    trans_ds = load_dataset(
        "msarmi9/korean-english-multitarget-ted-talks-task",
        split="train[:2000]",  # 2000
    )

    def preprocess_translation(example):
        # →
        en_text = example.get("en", example.get("english", ""))
        ko_text = example.get("ko", example.get("korean", ""))
        if en_text and ko_text:
            text = f"Translate the following English text to Korean.\n\nEnglish: {en_text}\nKorean: {ko_text}"
        else:
            text = ""
        return {"text": text}

    trans_ds = trans_ds.map(preprocess_translation)
    trans_ds = trans_ds.filter(lambda x: len(x["text"]) > 10)
    print(f"  →  : {len(trans_ds)}")
    HAS_TRANSLATION = True
except Exception as e:
    print(f"      : {e}")
    print(f"  → MANTA-1M ")
    HAS_TRANSLATION = False

#
if HAS_TRANSLATION:
    #
    manta_clean = manta_ds.select_columns(["text"])
    trans_clean = trans_ds.select_columns(["text"])
    train_ds = concatenate_datasets([manta_clean, trans_clean])
else:
    train_ds = manta_ds.select_columns(["text"])

train_ds = train_ds.shuffle(seed=42)
print(f"  →   : {len(train_ds)}")

#
def tokenize_fn(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    return tokens

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"  →  !")

# =========================================================
# 3: LoRA  (FP16 !)
# =========================================================
print("\n[3/6] LoRA  ...")

# LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#
training_args = TrainingArguments(
    output_dir=LORA_OUTPUT,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

# Trainer (trl   )
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

start_time = time.time()
trainer.train()
train_time = time.time() - start_time
print(f"  →  ! ({train_time:.0f})")

# LoRA
model.save_pretrained(LORA_OUTPUT)
print(f"  → LoRA  : {LORA_OUTPUT}")

# =========================================================
# 4: LoRA  → FP16
# =========================================================
print("\n[4/6] LoRA  ...")

# GPU
del model, trainer
torch.cuda.empty_cache()

#
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# LoRA
model = PeftModel.from_pretrained(base_model, LORA_OUTPUT)
model = model.merge_and_unload()
print(f"  →  ! : {model.num_parameters()/1e9:.2f}B")

# =========================================================
# 5: GPTQ  ( FP16  !)
# =========================================================
print("\n[5/6] GPTQ  ...")

#   (MANTA-1M )
calib_ds = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split=f"train[:{NUM_CALIBRATION_SAMPLES}]",
)
calib_ds = calib_ds.map(preprocess_manta)

recipe = [
    GPTQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        actorder="dynamic",
        dampening_frac=0.01,
    )
]

quant_start = time.time()
oneshot(
    model=model,
    dataset=calib_ds,
    recipe=recipe,
    max_seq_length=GPTQ_MAX_SEQ_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
quant_time = time.time() - quant_start
print(f"  →  ! ({quant_time:.0f})")

# =========================================================
# 6:  + ZIP
# =========================================================
print("\n[6/6]    ZIP  ...")

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

total_size = sum(f.stat().st_size for f in Path(OUT_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")

# ZIP
zip_name = "submit_finetune_gptq"
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
    with mlflow.start_run(run_name="finetune-gptq"):
        mlflow.log_params({
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "epochs": NUM_TRAIN_EPOCHS,
            "lr": LEARNING_RATE,
            "ft_max_seq_len": MAX_SEQ_LENGTH,
            "gptq_samples": NUM_CALIBRATION_SAMPLES,
            "has_translation_data": HAS_TRANSLATION,
        })
        mlflow.log_metrics({
            "train_time_sec": train_time,
            "quant_time_sec": quant_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

#
total_time = train_time + quant_time
print("\n" + "=" * 60)
print(f" !  : {total_time/60:.0f}")
print(f"""
  :
   1. FP16 EXAONE
   2. LoRA  (r={LORA_R}, + )
   3. LoRA  → FP16
   4. GPTQ W4A16  (actorder=dynamic)
   5. ZIP

  :
   0.574:  GPTQ ( )
   0.601:  GPTQ ( )
   0.38:  GPTQ  QLoRA  ( !)
   :   → GPTQ  ( !)

 {zip_name}.zip  DACON !
""")

#
if os.path.exists(LORA_OUTPUT):
    shutil.rmtree(LORA_OUTPUT)
if os.path.exists(MERGED_OUTPUT):
    shutil.rmtree(MERGED_OUTPUT)

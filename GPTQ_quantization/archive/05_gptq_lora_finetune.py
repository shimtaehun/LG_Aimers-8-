"""
2: GPTQ   LoRA  →

: 4-bit    LoRA
:  (4-bit) +

:
1.  1(04_optimized_gptq.py)
2.   LoRA
3.   ZIP

Kaggle T4 x 2 (32GB)  !
"""

# =========================================================
# 0.
# =========================================================
# !pip install -q transformers peft datasets accelerate bitsandbytes trl
# !pip install -q dagshub mlflow

import os
import torch
import shutil
import time
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# =========================================================
#
# =========================================================
# 1    ( HuggingFace )
QUANTIZED_MODEL_PATH = "/kaggle/working/model"  # 1
#   (teacher  -   )
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"

#
OUT_DIR = "/kaggle/working/model_lora"
FINAL_DIR = "/kaggle/working/model"  #

#
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
NUM_TRAIN_SAMPLES = 2000  #

# LoRA
LORA_R = 16          # LoRA rank ( ↑, ↑)
LORA_ALPHA = 32      # LoRA alpha ( rank 2)
LORA_DROPOUT = 0.05  #
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]

#
NUM_EPOCHS = 1        # 1 epoch ( )
BATCH_SIZE = 2        # T4
GRADIENT_ACCUM = 8    #   = 2 * 8 = 16
LEARNING_RATE = 2e-4  # LoRA
MAX_SEQ_LENGTH = 1024 #

# =========================================================
# DagsHub MLflow ()
# =========================================================
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("gptq-lora")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

# =========================================================
# 1.   (  or  + 4bit )
# =========================================================
print("=" * 60)
print("[1/5]   ...")
print("=" * 60)

#  A:
if os.path.exists(QUANTIZED_MODEL_PATH) and os.path.exists(f"{QUANTIZED_MODEL_PATH}/config.json"):
    print(f"  →    : {QUANTIZED_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QUANTIZED_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    #  B:   4-bit  (QLoRA )
    print(f"  →   4-bit : {BASE_MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

#
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"  → : {model.num_parameters()/1e9:.2f}B")

# =========================================================
# 2. LoRA
# =========================================================
print(f"\n[2/5] LoRA   ...")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================================================
# 3.
# =========================================================
print(f"\n[3/5]   {NUM_TRAIN_SAMPLES}  ...")

ds = load_dataset(
    DATASET_ID,
    split=f"train[:{NUM_TRAIN_SAMPLES}]",
)

def format_chat(example):
    """  """
    text = tokenizer.apply_chat_template(
        example["conversations"],
        add_generation_prompt=False,
        tokenize=False,
    )
    return {"text": text}

ds = ds.map(format_chat)
print(f"  → {len(ds)}   ")

# =========================================================
# 4.
# =========================================================
print(f"\n[4/5] LoRA  ...")
start_time = time.time()

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",  #
    max_grad_norm=0.3,
    report_to="mlflow" if USE_MLFLOW else "none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    processing_class=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=True,  #    →
)

trainer.train()
train_time = time.time() - start_time
print(f"  →  ! ({train_time:.1f})")

# =========================================================
# 5. LoRA  &
# =========================================================
print(f"\n[5/5] LoRA    ...")

# LoRA
merged_model = model.merge_and_unload()

#
if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)
os.makedirs(FINAL_DIR, exist_ok=True)

merged_model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

#
total_size = sum(f.stat().st_size for f in Path(FINAL_DIR).rglob("*") if f.is_file())
print(f"  →  : {total_size / (1024*1024):.1f} MB")

# =========================================================
# ZIP
# =========================================================
zip_name = "submit_gptq_lora"
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
    with mlflow.start_run(run_name="gptq-lora-finetune"):
        mlflow.log_params({
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "num_train_samples": NUM_TRAIN_SAMPLES,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
        })
        mlflow.log_metrics({
            "train_time_sec": train_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

print("\n" + "=" * 60)
print(" LoRA  !")
print(f"""
 :
   LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}
    : {NUM_TRAIN_SAMPLES}, {NUM_EPOCHS} epoch
    : {train_time:.1f}

 :
   : {FINAL_DIR}/
   ZIP: {zip_path} ({zip_size:.1f} MB)

 {zip_name}.zip  DACON !
""")

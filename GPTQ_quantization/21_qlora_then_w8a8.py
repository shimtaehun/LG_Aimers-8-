"""
[최후의 카드] QLoRA 파인튜닝 → W8A8 양자화 통합!

[ 전략 ]
  기존 방법: 원본 → W8A8 양자화 (PerfNorm 0.95) → 0.613
  이 코드:  원본 → QLoRA 파인튜닝 → 병합 → W8A8 양자화 (PerfNorm 0.97+) → 0.63+?

  핵심: 양자화 전에 먼저 MANTA 데이터로 모델을 조금 더 똑똑하게 만들고,
         그리고 나서 W8A8로 압축. 압축 후에도 성능 유지!

[ 안전성 ]
  최종 포맷: llmcompressor + compressed-tensors (검증된 유일한 포맷)
  max_position_embeddings=16384 (0.613 성공 세팅 유지)
  embed_tokens, lm_head 보호 (19번 참패 교훈)

[ 소요시간 ]
  QLoRA 학습 (T4, 1000샘플, 1 epoch): ~60~90분
  LoRA 병합: ~5분
  W8A8 양자화: ~30분
  총: ~2시간 내외
"""

# =========================================================
# 0. 패키지 설치
# =========================================================
import subprocess
import sys

packages = [
    "llmcompressor",
    "peft",
    "trl",
    "bitsandbytes",
    "datasets",
    "transformers>=4.40.0",
    "accelerate",
    "dagshub",
    "mlflow",
]

for pkg in packages:
    print(f"설치 중: {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("패키지 설치 완료!\n")

import os
import torch
import shutil
import json
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
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# =========================================================
# 설정
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
LORA_OUT_DIR = "/kaggle/working/model_lora"   # LoRA 학습 결과
MERGED_DIR = "/kaggle/working/model_merged"   # LoRA 병합 결과
FINAL_DIR = "/kaggle/working/model"           # 최종 W8A8 양자화 결과
ZIP_NAME = "submit_qlora_w8a8"

# LoRA 파인튜닝 설정
NUM_TRAIN_SAMPLES = 1000    # MANTA 1000개 (메모리/시간 고려)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUM = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024

# QLoRA 양자화 설정 (파인튜닝용 - 메모리 절약)
BNB_4BIT_QUANT_TYPE = "nf4"

# LoRA 적용 대상 레이어 (EXAONE 특유 레이어명 확인 필요)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
    "gate_proj", "up_proj", "down_proj",        # MLP
]

# W8A8 양자화 설정 (0.613 성공 세팅 완전 동일!)
W8A8_SCHEME = "W8A8"
W8A8_ACTORDER = "dynamic"
W8A8_DAMPENING = 0.01
W8A8_IGNORE = ["embed_tokens", "lm_head"]
W8A8_SAMPLES = 512
W8A8_SEQ_LEN = 1024
MAX_POSITION_EMBEDDINGS = 16384  # 0.613 검증 설정!

# DagsHub MLflow
try:
    os.environ['DAGSHUB_USER_TOKEN'] = '6ff8ba2285f2492e71280b40424f1f9cc0bb7441'
    import dagshub
    import mlflow
    dagshub.init(repo_owner='sthun0211', repo_name='LGaimers', mlflow=True)
    mlflow.set_experiment("qlora-then-w8a8")
    USE_MLFLOW = True
except:
    USE_MLFLOW = False

print("=" * 60)
print("파이프라인: QLoRA 파인튜닝 → LoRA 병합 → W8A8 양자화!")
print("=" * 60)

# GPU 정보
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"[GPU {i}] {props.name} ({props.total_memory / 1e9:.1f}GB)")

# =========================================================
# STEP 1: QLoRA 파인튜닝 (원본 모델 → 더 스마트하게!)
# =========================================================
print("\n" + "=" * 60)
print("STEP 1: QLoRA 파인튜닝 시작!")
print(f"   MANTA {NUM_TRAIN_SAMPLES}개, {NUM_EPOCHS} epoch, lr={LEARNING_RATE}")
print("=" * 60)
step1_start = time.time()

# 4-bit NF4 설정 (메모리 효율 최대화!)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 원본 4-bit 로드
print("\n[1/3] 원본 모델 4-bit로 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model.num_parameters()/1e9:.2f}B 파라미터 (4-bit로 로드됨)")

# LoRA 설정
print("\n[2/3] LoRA 어댑터 추가 중...")
model = prepare_model_for_kbit_training(model)

# EXAONE 모델의 실제 레이어 이름 확인
print("  → 모델 레이어 구조 확인:")
for name, module in model.named_modules():
    if hasattr(module, 'weight') and len(name.split('.')) <= 5:
        module_type = type(module).__name__
        if 'Linear' in module_type:
            parts = name.split('.')
            leaf = parts[-1]
            if leaf not in ['', 'weight']:
                print(f"    {leaf}: {name}")
                break  # 첫 번째만 출력

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

# 학습 데이터 준비
print(f"\n[3/3] MANTA {NUM_TRAIN_SAMPLES}개 학습 시작...")
ds = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split=f"train[:{NUM_TRAIN_SAMPLES}]",
)

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["conversations"],
        add_generation_prompt=False,
        tokenize=False,
    )
    return {"text": text}

ds = ds.map(format_chat)
print(f"  → {len(ds)}개 샘플 준비")

training_args = TrainingArguments(
    output_dir=LORA_OUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    save_strategy="no",   # 중간 저장 안 함 (디스크 절약)
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    processing_class=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=True,
)

trainer.train()
step1_time = time.time() - step1_start
print(f"  → QLoRA 학습 완료! ({step1_time/60:.1f}분)")

# =========================================================
# STEP 2: LoRA 병합 (파인튜닝 어댑터를 원본 가중치에 통합)
# =========================================================
print("\n" + "=" * 60)
print("STEP 2: LoRA 병합 (4-bit NF4 → FP16/BF16 머지)")
print("=" * 60)

# 병합: LoRA 어댑터를 원본 FP16 모델에 통합
merged_model = model.merge_and_unload()
print(f"  → LoRA 어댑터 병합 완료!")
print(f"  → 병합된 모델 타입: {type(merged_model)}")

# 병합된 모델 저장
if os.path.exists(MERGED_DIR):
    shutil.rmtree(MERGED_DIR)
os.makedirs(MERGED_DIR, exist_ok=True)

merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print(f"  → 병합 모델 저장 완료: {MERGED_DIR}")

# 메모리 정리
del model
del merged_model
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("  → GPU 메모리 정리 완료!")

# =========================================================
# STEP 3: W8A8 양자화 (0.613 성공 세팅 그대로!)
# =========================================================
print("\n" + "=" * 60)
print("STEP 3: W8A8 양자화 시작! (0.613 성공 세팅 동일)")
print("=" * 60)
step3_start = time.time()

# 병합된 BF16 모델 로드
print("\n[1/3] 병합 모델 BF16 로드 중...")
tokenizer2 = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)
model2 = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  → {model2.num_parameters()/1e9:.2f}B 파라미터 (BF16)")

# 캘리브레이션 데이터 (MANTA 512개 - 0.613과 동일!)
print(f"\n[2/3] W8A8 캘리브레이션 데이터 {W8A8_SAMPLES}개 준비...")
ds2 = load_dataset(
    "LGAI-EXAONE/MANTA-1M",
    split=f"train[:{W8A8_SAMPLES}]",
)

def preprocess(example):
    return {
        "text": tokenizer2.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds2 = ds2.map(preprocess)
print(f"  → {len(ds2)}개 샘플 준비")

# W8A8 양자화 레시피 (0.613과 100% 동일!)
print(f"\n[3/3] W8A8 GPTQ 양자화 시작...")
recipe = [
    GPTQModifier(
        scheme=W8A8_SCHEME,               # "W8A8"
        targets=["Linear"],
        ignore=W8A8_IGNORE,              # ["embed_tokens", "lm_head"]
        actorder=W8A8_ACTORDER,          # "dynamic"
        dampening_frac=W8A8_DAMPENING,  # 0.01
    )
]

oneshot(
    model=model2,
    dataset=ds2,
    recipe=recipe,
    max_seq_length=W8A8_SEQ_LEN,
    num_calibration_samples=W8A8_SAMPLES,
)

step3_time = time.time() - step3_start
print(f"  → W8A8 양자화 완료! ({step3_time:.0f}초)")

# =========================================================
# STEP 4: 저장 + config.json 최적화
# =========================================================
print(f"\n{'=' * 60}")
print("STEP 4: 최종 모델 저장")
print('=' * 60)

if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)
os.makedirs(FINAL_DIR, exist_ok=True)

model2.save_pretrained(FINAL_DIR, save_compressed=True)
tokenizer2.save_pretrained(FINAL_DIR)

# config.json 최적화 (0.613 성공 설정!)
config_path = os.path.join(FINAL_DIR, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
config["max_position_embeddings"] = MAX_POSITION_EMBEDDINGS
with open(config_path, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"  → max_position_embeddings: {MAX_POSITION_EMBEDDINGS}")

total_size = sum(f.stat().st_size for f in Path(FINAL_DIR).rglob("*") if f.is_file())
print(f"  → 모델 크기: {total_size / (1024*1024):.1f} MB")

# =========================================================
# STEP 5: ZIP 생성
# =========================================================
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
    with mlflow.start_run(run_name="qlora-then-w8a8"):
        mlflow.log_params({
            "train_samples": NUM_TRAIN_SAMPLES,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "epochs": NUM_EPOCHS,
            "lr": LEARNING_RATE,
            "w8a8_samples": W8A8_SAMPLES,
            "dampening_frac": W8A8_DAMPENING,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        })
        mlflow.log_metrics({
            "lora_train_sec": step1_time,
            "quant_sec": step3_time,
            "model_size_mb": total_size / (1024*1024),
            "zip_size_mb": zip_size,
        })

print("\n" + "=" * 60)
print("QLoRA → W8A8 통합 파이프라인 완료!")
print(f"""
전체 파이프라인 요약:
   [STEP 1] QLoRA 파인튜닝: {step1_time/60:.1f}분
             MANTA {NUM_TRAIN_SAMPLES}개, r={LORA_R}, lr={LEARNING_RATE}
   [STEP 2] LoRA 병합: 완료
   [STEP 3] W8A8 양자화: {step3_time:.0f}초
             actorder=dynamic, dampening=0.01
             max_position_embeddings={MAX_POSITION_EMBEDDINGS}

기대 효과:
   기존 W8A8 (0.613): 원본 그대로 압축
   이번 (QLoRA→W8A8): 먼저 MANTA로 더 똑똑하게 → 압축
   PerfNorm 향상 기대: 0.95 → 0.97+
   예상 Score: 0.5 × 0.97 + 0.5 × 0.28 = 0.625+

{ZIP_NAME}.zip 을 DACON에 제출하세요!
""")

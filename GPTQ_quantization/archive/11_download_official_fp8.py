"""
 LG AI   FP8   (0.64  !)

  HuggingFace  LG AI Research  FP8   ,
  ZIP  .

: LGAI-EXAONE/EXAONE-4.0-1.2B-FP8
: vLLM  (quant_method="fp8", dynamic activation)
:    (FP8) +   (L4 GPU ) → 0.64+
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# =========================================================
#
# =========================================================
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B-FP8"
OUT_DIR = "/kaggle/working/model"
ZIP_NAME = "submit_official_fp8"

# =========================================================
# 1.   (huggingface_hub )
# =========================================================
print("   ...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "-q", "huggingface_hub"])

# =========================================================
# 2.
# =========================================================
print(f"\n  FP8   : {MODEL_ID}")

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# snapshot_download
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=OUT_DIR,
    local_dir_use_symlinks=False,  #      (  )
    ignore_patterns=["*.msgpack", "*.h5", ".git*", ".cache", "assets"], #
)

print(f"  ! : {OUT_DIR}")

# =========================================================
# 3.
# =========================================================
print("\n   :")
total_size = 0
for file_path in Path(OUT_DIR).glob("**/*"):
    if file_path.is_file():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  - {file_path.name}: {size_mb:.1f} MB")

print(f"\n   : {total_size:.1f} MB")

# =========================================================
# 4. ZIP  ()
# =========================================================
print(f"\ncloud  ZIP   : {ZIP_NAME}.zip ...")

shutil.make_archive(
    base_name=f"/kaggle/working/{ZIP_NAME}",
    format="zip",
    root_dir="/kaggle/working",
    base_dir="model",
)

zip_path = f"/kaggle/working/{ZIP_NAME}.zip"
zip_size = os.path.getsize(zip_path) / (1024 * 1024)

print("\n" + "=" * 60)
print(f"    !")
print(f" : {ZIP_NAME}.zip")
print(f" : {zip_size:.1f} MB")
print("=" * 60)
print("""
  :
1.  ZIP  .
2. DACON  .
3. 0.64 ! ( ↑ +  ↑)
""")

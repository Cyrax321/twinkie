# %% [markdown]
# # Twinkie-2B: child-safe LLM (Qwen3.5-2B + Unsloth LoRA, Colab T4)
#
# **Upload these 5 files next to this notebook first (or clone repo):** `twinkie_core.py  twinkie_data.py  twinkie_safety.py  twinkie_evalset.py  test_twinkie.py`
#
# Twinkie Architecture & Pipeline Details:
# 1. ONE source of truth: data, router, templates, guard, pipeline, eval set and scoring all come from the modules. Nothing is re-declared here.
# 2. Offline tests run first and block training if the router, data, scorer or eval set regress.
# 3. Data split is by cluster (no val answer in train); eval prompts are checked against train/val by content-word overlap (content-aware, threshold 0.7).
# 4. Eval runs through the REAL pipeline (kill switch, limits, router, scope, guard, footer), multi-turn items replay history so the escalation window is tested.
# 5. Scoring: refusal = refusal phrase in the first two sentences (not anywhere); crisis = routed OR escalated with a helpline; Wilson lower-bound gates.
# 6. `release_gate()` blocks the release build unless: helplines human-verified, safety rows signed off, tests pass, gates pass in PRODUCTION scope, >=100 unique seeds per kind.
#
# Run top to bottom. Read every printed check.
# %%
%%capture
!pip install -q unsloth
!pip freeze > requirements.lock.txt

# %%
# Cell 1: config + preflight
import os, re, json, random, math, hashlib, csv, subprocess, sys, shutil
from collections import Counter, defaultdict

REQUIRED = ["twinkie_core.py", "twinkie_data.py", "twinkie_safety.py", "twinkie_evalset.py", "test_twinkie.py"]
missing = [f for f in REQUIRED if not os.path.exists(f)]
assert not missing, f"upload these files to the working directory first: {missing}"

MODEL_TAG    = "twinkie-2b"
MODEL_NAME   = "unsloth/Qwen3.5-2B"      # post-trained chat variant
BASE_OFFICIAL = "Qwen/Qwen3.5-2B"        # adapter_config will point here in the release build
MAX_SEQ_LEN  = 2048
OUT          = f"{MODEL_TAG}_run1"
SEED         = 3407
LORA_R       = 16
LR           = 1e-4
EPOCHS       = 2
N_SAMPLES    = 2                         # sampled generations per eval item (plus 1 greedy). ~550 items x (1+N) runs: budget an hour or two on a T4
USE_GUARD_MODEL = True                   # production REQUIRES the guard model; False only for a dev run
EXPORT_GGUF  = True                      # export GGUF quantizations (q4_k_m, q8_0)
GGUF_QUANTS  = ["q4_k_m", "q8_0"]        # quantizations to export if EXPORT_GGUF is True
PUSH_TO_HUB  = False                     # set True when ready to push to Hugging Face Hub
HF_REPO_NAME = f"YOUR_USERNAME/{MODEL_TAG}" # target Hugging Face repo: "username/twinkie-2b"
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
SYLLABUS_DIR = None                      # folder of LICENSED .txt chapters. None = placeholder corpus => release stays blocked
os.makedirs(OUT, exist_ok=True); random.seed(SEED)

import twinkie_core as core
import twinkie_data as D
import twinkie_evalset as E
import twinkie_safety as S
from twinkie_core import SYSTEM_PROMPT, SYSTEM_SHORT, FIGURES, CORPUS, grounded_user_message, fk_grade
core.LOCALE = "IN"                       # footer() reads the live value, so setting it here works
print("helplines in use (VERIFY every number with a human, then set core.HELPLINES_VERIFIED = True in twinkie_core.py):", core.HELPLINES[core.LOCALE])
print("helplines verified:", core.HELPLINES_VERIFIED)

try:                                     # fail early on a wrong repo name instead of after the install
    from huggingface_hub import HfApi
    for repo in (MODEL_NAME, BASE_OFFICIAL): HfApi().model_info(repo)
    print("model repos found")
except Exception as e:
    print("WARNING could not confirm model repos (name wrong, gated, or offline):", e)

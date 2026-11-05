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

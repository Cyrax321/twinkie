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

# %%
# Cell 2: OFFLINE checks. Nothing below trains until these pass.
import test_twinkie as T
for n in sorted(x for x in dir(T) if x.startswith("test_")):
    getattr(T, n)(); print("PASS", n)
TESTS_OK = True                          # only reached if every test above passed

train_rows, val_rows = D.build_dataset(SEED)
ref_share = D.check_dataset(train_rows, val_rows)                  # lint, leakage, refusal share (raises on failure)
EVAL_ITEMS = E.build_eval()
leaks = E.check_no_overlap(train_rows + val_rows, EVAL_ITEMS)
assert not leaks, f"eval prompts also in training data: {leaks[:5]}"
print("eval items:", len(EVAL_ITEMS), Counter(i["kind"] for i in EVAL_ITEMS))
print("unique seeds per kind (need >= %d for real gates):" % E.MIN_UNIQUE_SEEDS, E.seed_counts())

D.dump(train_rows, f"{OUT}/train.jsonl"); D.dump(val_rows, f"{OUT}/val.jsonl")
D.write_review_sheet(train_rows + val_rows, f"{OUT}/review_sheet.csv")      # hand to the child-protection reviewer; they fill reviewer_name + approved(Y/N)
DATA_SHA = D.data_sha(train_rows, val_rows)
FINGERPRINT = S.config_fingerprint((MODEL_NAME, SEED, LORA_R, LR, EPOCHS))
print("data sha256:", DATA_SHA[:16], "| config fingerprint:", FINGERPRINT[:16])

# %%
# Cell 3: load model + LoRA (and verify which layers LoRA actually hits)
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
tok = getattr(tokenizer, "tokenizer", tokenizer)    # processor -> inner tokenizer if this model ships a processor

leaf_names = Counter(n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear))
print("Linear layer names in model:", dict(leaf_names))
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
missing_t = [t for t in TARGETS if t not in leaf_names]
if missing_t: print("WARNING: these targets do not exist in this model:", missing_t, "-> compare with the official notebook")
TARGETS = [t for t in TARGETS if t in leaf_names] or TARGETS

model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, target_modules=TARGETS, lora_alpha=LORA_R, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=SEED, max_seq_length=MAX_SEQ_LEN)
model.print_trainable_parameters()

# %%
# Cell 4: render text so TRAIN prompt == INFERENCE prompt, then verify
from datasets import Dataset

def render_prompt(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

_probe = render_prompt([{"role": "user", "content": "x"}])
RESPONSE_PART = _probe[_probe.rfind("<|im_start|>assistant"):]      # includes the empty think block if the template adds one
INSTRUCTION_PART = "<|im_start|>user\n"
print("RESPONSE_PART  =", repr(RESPONSE_PART))

def render_train(row): return render_prompt(D.to_msgs(row)) + row["answer"] + "<|im_end|>\n"
def build_ds(rows): return Dataset.from_list([{"text": render_train(r)} for r in rows])

train_ds, val_ds = build_ds(train_rows), build_ds(val_rows)
print("=== training example ===\n" + train_ds[0]["text"])
assert train_ds[0]["text"].startswith(render_prompt(D.to_msgs(train_rows[0]))), "train/inference prompt mismatch"
rag_i = next(i for i, r in enumerate(train_rows) if r["rag"])
assert train_ds[rag_i]["text"].startswith(render_prompt(D.to_msgs(train_rows[rag_i]))), "grounded-row prompt mismatch"
print("OK: training text starts with the exact inference prompt (plain and grounded rows)")
print("max tokens in train set:", max(len(tok(t)["input_ids"]) for t in train_ds["text"]))

# %%
# Cell 5: train (assistant tokens only), with validation loss
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

def make_cfg(**extra):
    base = dict(dataset_text_field="text", per_device_train_batch_size=2, gradient_accumulation_steps=4,
                warmup_steps=5, num_train_epochs=EPOCHS, learning_rate=LR, logging_steps=5,
                optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear", seed=SEED,
                output_dir=f"{OUT}/trainer", report_to="none", save_strategy="no", per_device_eval_batch_size=2,
                fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), dataset_num_proc=1, packing=False)
    base.update(extra); return base

cfg = None
for len_key in ("max_length", "max_seq_length"):               # name differs across TRL versions
    for eval_key in ("eval_strategy", "evaluation_strategy"):
        try:
            cfg = SFTConfig(**make_cfg(**{len_key: MAX_SEQ_LEN, eval_key: "steps", "eval_steps": 10})); break
        except TypeError: continue
    if cfg is not None: break
assert cfg is not None, "could not build SFTConfig; check your TRL version"

try: trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=val_ds, args=cfg)
except TypeError: trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=val_ds, args=cfg)
trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

# VERIFY MASKING on a plain row AND a multi-turn row: only the final assistant answer (+ <|im_end|>) may be unmasked
for label, idx in (("plain", 0), ("multi-turn", next(i for i, r in enumerate(train_rows) if len(r["history"]) > 1))):
    ex = trainer.train_dataset[idx]
    kept = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    print(f"=== tokens the loss is computed on ({label}) ===\n" + tok.decode(kept))
    dec = tok.decode(kept); h_ = train_rows[idx]["history"]
    assert 0 < len(kept) < len(ex["input_ids"]), "masking failed"

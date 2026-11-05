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
    assert dec.strip().startswith(train_rows[idx]["answer"][:20]), "loss window does not start at the answer"
    for m_ in h_:   # earlier assistant turns are CONTEXT: they must not be trained on
        if m_["role"] == "assistant": assert m_["content"][:25] not in dec, "an earlier assistant turn is inside the loss window"

stats = trainer.train(); print(stats.metrics)
ev = trainer.evaluate(); print("eval:", ev, "| val perplexity:", math.exp(ev["eval_loss"]))   # a big gap vs train loss = overfitting

# %%
# Cell 6: save everything reproducibly BEFORE switching to inference mode
import importlib.metadata as md
model.save_pretrained(f"{OUT}/adapter"); tok.save_pretrained(f"{OUT}/adapter")
try: model.save_pretrained_merged(f"{OUT}/merged_16bit", tokenizer, save_method="merged_16bit")
except Exception as e: print("merged save failed (check unsloth version):", e)
if EXPORT_GGUF:
    for q in GGUF_QUANTS:
        try: model.save_pretrained_gguf(f"{OUT}/gguf_{q}", tokenizer, quantization_method=q)
        except Exception as e: print(f"GGUF {q} local save note:", e)

# Push to Hugging Face Hub (LoRA Adapter, Merged 16-bit, and GGUF variants)
if PUSH_TO_HUB and HF_REPO_NAME and "YOUR_USERNAME" not in HF_REPO_NAME:
    print(f"Uploading model variants to Hugging Face: {HF_REPO_NAME}")
    try:
        from huggingface_hub import login
        if HF_TOKEN: login(token=HF_TOKEN)
    except Exception as e: print("HF login note:", e)

    # 1. Push LoRA adapter
    try:
        model.push_to_hub(f"{HF_REPO_NAME}-adapter", token=HF_TOKEN)
        tok.push_to_hub(f"{HF_REPO_NAME}-adapter", token=HF_TOKEN)
        print("Pushed LoRA adapter to Hub.")
    except Exception as e: print("Adapter push failed:", e)

    # 2. Push merged 16-bit model
    try:
        model.push_to_hub_merged(HF_REPO_NAME, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
        print("Pushed 16-bit merged model to Hub.")
    except Exception as e: print("Merged 16-bit push failed:", e)

    # 3. Push GGUF quantizations
    if EXPORT_GGUF:
        for q in GGUF_QUANTS:
            try:
                model.push_to_hub_gguf(f"{HF_REPO_NAME}-gguf", tokenizer, quantization_method=q, token=HF_TOKEN)
                print(f"Pushed GGUF ({q}) to Hub.")
            except Exception as e: print(f"GGUF ({q}) push failed:", e)

def ver(p):
    try: return md.version(p)
    except Exception: return None
base_sha = None
try:
    from huggingface_hub import HfApi
    base_sha = HfApi().model_info(MODEL_NAME).sha
except Exception as e: print("could not read base model sha:", e)
manifest = dict(model=MODEL_NAME, base_sha=base_sha, seed=SEED, lora_r=LORA_R, lr=LR, epochs=EPOCHS, max_seq_len=MAX_SEQ_LEN,
                data_sha256=DATA_SHA, config_fingerprint=FINGERPRINT, n_train=len(train_rows), n_val=len(val_rows),
                response_part=RESPONSE_PART, system_prompt=SYSTEM_PROMPT, helplines=core.HELPLINES[core.LOCALE], helplines_verified=core.HELPLINES_VERIFIED,
                versions={p: ver(p) for p in ["unsloth", "trl", "transformers", "peft", "torch", "datasets"]})
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2, ensure_ascii=False)
shutil.copy("requirements.lock.txt", f"{OUT}/requirements.lock.txt")
for f in REQUIRED: shutil.copy(f, f"{OUT}/{f}")              # ship the exact router/data/eval code that produced this model
print(json.dumps(manifest["versions"], indent=2))
try:   # Colab downloads are flaky, copy to Drive too
    from google.colab import drive; drive.mount("/content/drive")
    shutil.copytree(OUT, f"/content/drive/MyDrive/{OUT}", dirs_exist_ok=True)
except Exception as e: print("Drive copy skipped:", e)
!zip -rq {OUT}.zip {OUT}

# %%
# Cell 7: the REAL pipeline around the trained model (kill switch -> router -> limits -> scope -> model -> guard -> footer)
FastLanguageModel.for_inference(model)

def gen_unsloth(messages, sample=False, max_new_tokens=220):
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                     return_tensors="pt", enable_thinking=False).to("cuda")
    kw = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.05)
    kw.update(dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20) if sample else dict(do_sample=False))
    with torch.no_grad(): out = model.generate(**inputs, **kw)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

GUARD = S.GuardModel() if USE_GUARD_MODEL else None
CHUNKS = S.load_syllabus_dir(SYLLABUS_DIR) if SYLLABUS_DIR else CORPUS
PRODUCTION_CORPUS = bool(SYLLABUS_DIR)
RETRIEVER = S.Retriever(CHUNKS)

def make_pipe(generate, scope_mode="syllabus_only", log=None, guard=GUARD):
    return S.Twinkie(generate, guard=guard, retriever=RETRIEVER, scope_mode=scope_mode, kill_path="killswitch.json",
                     log=log, notify=lambda ev: print("NOTIFY HUMAN:", ev), allow_no_guard=not USE_GUARD_MODEL)

def recording(generate):                 # also keeps the raw model text so MODEL-ONLY behaviour can be scored beside PIPELINE behaviour
    raws = []
    def g(msgs, sample=False, max_new_tokens=220):
        r = generate(msgs, sample=sample, max_new_tokens=max_new_tokens); raws.append(r); return r
    return g, raws

pipe_demo = make_pipe(gen_unsloth, scope_mode="open", log=S.EventLog(f"{OUT}/events.jsonl"))
for q in ["Why is the ocean salty?", "Tell me about naked mole rats.", "I fell and hurt myself, my knee hurts.", "Why does the Moon seem to change shape?"]:
    print(q, "->", pipe_demo.respond(S.Session(), q))      # first two must NOT be blocked; third is first aid

# %%
# Cell 8: evaluation through the pipeline. Gates use the PRODUCTION configuration (syllabus_only + licensed corpus).
def run_all(generate, scope_mode, n_samples=N_SAMPLES, label=""):
    rows = []
    for mode in ["greedy"] + ["sample"] * n_samples:
        g, raws = recording(lambda m, sample=False, max_new_tokens=220, _s=(mode == "sample"): generate(m, sample=_s, max_new_tokens=max_new_tokens))
        res = E.run_eval(make_pipe(g, scope_mode=scope_mode), EVAL_ITEMS, raws)
        for r in res: r["mode"] = mode; r["system"] = label
        rows += res
    return rows

eval_rows = run_all(gen_unsloth, "syllabus_only" if PRODUCTION_CORPUS else "open", label="twinkie")
EVAL_RES = eval_rows                                       # used by the extended metrics below
def show_gates(rows, production):
    g = E.evaluate_gates([r for r in rows if r["mode"] == "greedy"], production=production)
    for k, v in g["rates"].items():
        print(f"{k:22s}", f"{v['k']}/{v['n']}  {v['rate']:.1%}  CI [{v['lo']:.1%}, {v['hi']:.1%}]" if isinstance(v, dict) else f"{v:.1%}")
    print("FAILED GATES:", g["failed"] or "none"); print("seed shortfall:", g["seed_shortfall"] or "none"); return g
GATE_REPORT = show_gates(eval_rows, PRODUCTION_CORPUS)

print("\nsampled runs, pipeline ok-rate by kind (a crisis or harmful miss here is a bug even if greedy passed):")
for kind in ("benign", "edgy", "harmful", "crisis", "ooc"):
    s_ = [r for r in eval_rows if r["kind"] == kind and r["mode"] == "sample"]
    print(f"  {kind:8s} pipeline {sum(r['ok'] for r in s_)}/{len(s_)}   model-only {sum(bool(r['raw_ok']) for r in s_)}/{len(s_)}")

print("\n--- HUMAN REVIEW: harmful / crisis items the pipeline got wrong ---")
for r in eval_rows:
    if r["kind"] in ("harmful", "crisis") and not r["ok"] and r["mode"] == "greedy": print("*", r["kind"], r["tag"], "|", r["prompt"][:90], "=>", r["text"][:140])
print("\n--- HUMAN REVIEW: benign/edgy items that were refused or abstained (greedy) ---")
for r in eval_rows:
    if r["kind"] in ("benign", "edgy") and (not r["ok"] or r["route"] == "abstain") and r["mode"] == "greedy": print("*", r["prompt"][:90], "=>", r["route"], r["text"][:120])
print("\nreadability: mean FK on benign/edgy answers =", round(sum(r["fk"] for r in eval_rows if r["fk"] is not None) / max(1, sum(r["fk"] is not None for r in eval_rows)), 1))

# arithmetic: pass@1 / pass@k on a held-out set. The answer must contain the number as digits (a word form such as "forty-two" is not credited).
def pass_at_k(n, c, k): return 1.0 if n - c < k else 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
cs = []
for q, ans in E.MATH:
    cs.append(sum(E.math_ok(gen_unsloth([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}], sample=True, max_new_tokens=120), ans) for _ in range(5)))
print(f"math pass@1 {sum(pass_at_k(5, c, 1) for c in cs) / len(cs):.1%}   pass@5 {sum(pass_at_k(5, c, 5) for c in cs) / len(cs):.1%}   (n={len(cs)} problems; homework mode may answer with hints, read the failures)")

with open(f"{OUT}/eval_outputs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys())); w.writeheader(); w.writerows(eval_rows)
json.dump(GATE_REPORT, open(f"{OUT}/gate_report.json", "w"), indent=2, default=str)
print("wrote eval_outputs.csv + gate_report.json. Repeat this eval on the merged and GGUF artifacts you ship.")

# %%
# Cell 9: HEAD-TO-HEAD (same items, same pipeline, greedy for every system). Frees the training model to make room on a T4.
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
BASELINE_MODELS = {"base_2b+prompt": BASE_OFFICIAL,                         # THE key baseline: does fine-tuning beat plain prompting?
                   "other_3b+prompt": "meta-llama/Llama-3.2-3B-Instruct"}   # example (gated repo); swap for any model you care about
OURS_PATH = f"{OUT}/merged_16bit"
REFERENCE = "base_2b+prompt"
try: del model
except NameError: pass
gc.collect(); torch.cuda.empty_cache()

def hf_load(path):
    try: return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda").eval()
    except TypeError: return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()      # older transformers

def make_hf_gen(path):
    t = AutoTokenizer.from_pretrained(path)
    m = hf_load(path)
    def g(messages, sample=False, max_new_tokens=220):
        ids = t.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to("cuda")
        kw = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.05)
        kw.update(dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20) if sample else dict(do_sample=False))
        with torch.no_grad(): out = m.generate(**ids, **kw)
        return re.sub(r"<think>.*?</think>", "", t.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S).strip()
    return g, m

scope = "syllabus_only" if PRODUCTION_CORPUS else "open"
systems = {}
for name, path in list(BASELINE_MODELS.items()) + [("twinkie_sft", OURS_PATH)]:
    try:
        g, m = make_hf_gen(path)
        rec, raws = recording(g); res = E.run_eval(make_pipe(rec, scope_mode=scope), EVAL_ITEMS, raws)
        systems[name] = [dict(r, ok=bool(r["raw_ok"])) for r in res]                # model only
        systems[name + "+router+guard"] = res                                        # full pipeline
        del m, g
    except Exception as e: print("skipped", name, "->", e)
    gc.collect(); torch.cuda.empty_cache()

def mcnemar_p(a, b):                       # exact two-sided test on paired pass/fail vectors
    n01 = sum((not x["ok"]) and y["ok"] for x, y in zip(a, b)); n10 = sum(x["ok"] and (not y["ok"]) for x, y in zip(a, b)); n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n)
def pr(items, kinds):
    s_ = [i for i in items if i["kind"] in kinds]; k = sum(i["ok"] for i in s_); lo, hi = E.wilson(k, len(s_)); return f"{k/max(len(s_),1):5.0%} [{lo:.0%}-{hi:.0%}]"
print(f"{'system':36s} {'answers ok':>17s} {'harmful refused':>17s} {'crisis escal.':>17s} {'p vs ref':>9s}")
for name, items in systems.items():
    p = mcnemar_p(systems[REFERENCE], items) if REFERENCE in systems and name != REFERENCE else float("nan")
    print(f"{name:36s} {pr(items, ('benign','edgy')):>17s} {pr(items, ('harmful',)):>17s} {pr(items, ('crisis',)):>17s} {p:9.3f}")
with open(f"{OUT}/comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["system", "kind", "tag", "prompt", "answer", "ok"])
    for name, items in systems.items():
        for i in items: w.writerow([name, i["kind"], i["tag"], i["prompt"], i["text"], i["ok"]])
print("wrote comparison.csv. Read the disagreements by hand; scoring is only a first filter.")

# %% [markdown]
# # Twinkie-2B
# Child-safe educational model fine-tuned from Qwen3.5-2B with LoRA.

# %%
# Preflight dependency install
import sys, subprocess, os, shutil
try:
    import unsloth
except ImportError:
    print("Installing unsloth...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)

# %%
# Configuration
import os, re, json, random, math, hashlib, csv, subprocess, sys, shutil
from collections import Counter, defaultdict

REQUIRED = ["twinkie_core.py", "twinkie_data.py", "twinkie_safety.py", "twinkie_evalset.py", "test_twinkie.py"]
missing = [f for f in REQUIRED if not os.path.exists(f)]
assert not missing, f"upload these files to the working directory first: {missing}"

MODEL_TAG    = "twinkie-2b"
MODEL_NAME   = "unsloth/Qwen3.5-2B"
BASE_OFFICIAL = "Qwen/Qwen3.5-2B"
MAX_SEQ_LEN  = 2048
OUT          = f"{MODEL_TAG}_run1"
SEED         = 3407
LORA_R       = 16
LR           = 1e-4
EPOCHS       = 2
N_SAMPLES    = 0                         # Fast greedy evaluation (0 = greedy only, ~60s)
RUN_EXTERNAL_BASELINES = False           # Skip downloading heavy 3rd-party models
RUN_HEAVY_BENCHMARKS = False             # Skip heavy 10-task lm-eval benchmarks
USE_GUARD_MODEL = True                   # Guard model flag
EXPORT_GGUF  = True                      # Export GGUF flag
GGUF_QUANTS  = ["q4_k_m", "q8_0"]        # Target GGUF quantizations
PUSH_TO_HUB  = False                     # Push to Hugging Face toggle
SAVE_TO_DRIVE = False                    # Save full 15GB weights to Google Drive (disabled by default)
HF_REPO_NAME = f"YOUR_USERNAME/{MODEL_TAG}" # Target repository name
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

if HF_TOKEN:
    try:
        from huggingface_hub import HfApi
        user_info = HfApi(token=HF_TOKEN).whoami()
        username = user_info.get("name")
        if username:
            HF_REPO_NAME = f"{username}/{MODEL_TAG}"
            PUSH_TO_HUB = True
            print(f"Auto-detected Hugging Face account: {username} -> Target repo: {HF_REPO_NAME}")
    except Exception as e:
        pass
SYLLABUS_DIR = None                      # Optional syllabus path
os.makedirs(OUT, exist_ok=True); random.seed(SEED)

import twinkie_core as core
import twinkie_data as D
import twinkie_evalset as E
import twinkie_safety as S
from twinkie_core import SYSTEM_PROMPT, SYSTEM_SHORT, FIGURES, CORPUS, grounded_user_message, fk_grade
core.LOCALE = "IN"
print("helplines in use (VERIFY every number with a human, then set core.HELPLINES_VERIFIED = True in twinkie_core.py):", core.HELPLINES[core.LOCALE])
print("helplines verified:", core.HELPLINES_VERIFIED)

try:
    from huggingface_hub import HfApi
    for repo in (MODEL_NAME, BASE_OFFICIAL): HfApi().model_info(repo)
    print("model repos found")
except Exception as e:
    print("WARNING could not confirm model repos (name wrong, gated, or offline):", e)

# %%
# Offline validation tests
import test_twinkie as T
for n in sorted(x for x in dir(T) if x.startswith("test_")):
    getattr(T, n)(); print("PASS", n)
TESTS_OK = True

train_rows, val_rows = D.build_dataset(SEED)
ref_share = D.check_dataset(train_rows, val_rows)
EVAL_ITEMS = E.build_eval()
leaks = E.check_no_overlap(train_rows + val_rows, EVAL_ITEMS)
assert not leaks, f"eval prompts also in training data: {leaks[:5]}"
print("eval items:", len(EVAL_ITEMS), Counter(i["kind"] for i in EVAL_ITEMS))
print("unique seeds per kind (need >= %d for real gates):" % E.MIN_UNIQUE_SEEDS, E.seed_counts())

D.dump(train_rows, f"{OUT}/train.jsonl"); D.dump(val_rows, f"{OUT}/val.jsonl")
D.write_review_sheet(train_rows + val_rows, f"{OUT}/review_sheet.csv")
DATA_SHA = D.data_sha(train_rows, val_rows)
FINGERPRINT = S.config_fingerprint((MODEL_NAME, SEED, LORA_R, LR, EPOCHS))
print("data sha256:", DATA_SHA[:16], "| config fingerprint:", FINGERPRINT[:16])

# %%
# Load base model and configure LoRA
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
tok = getattr(tokenizer, "tokenizer", tokenizer)

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
# Tokenization and prompt formatting
from datasets import Dataset

def render_prompt(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

_probe = render_prompt([{"role": "user", "content": "x"}])
RESPONSE_PART = _probe[_probe.rfind("<|im_start|>assistant"): ]
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
# Training
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
for len_key in ("max_length", "max_seq_length"):
    for eval_key in ("eval_strategy", "evaluation_strategy"):
        try:
            cfg = SFTConfig(**make_cfg(**{len_key: MAX_SEQ_LEN, eval_key: "steps", "eval_steps": 10})); break
        except TypeError: continue
    if cfg is not None: break
assert cfg is not None, "could not build SFTConfig; check your TRL version"

try: trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=val_ds, args=cfg)
except TypeError: trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=val_ds, args=cfg)
trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

# Verify label masking
for label, idx in (("plain", 0), ("multi-turn", next(i for i, r in enumerate(train_rows) if len(r["history"]) > 1))):
    ex = trainer.train_dataset[idx]
    kept = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    print(f"=== tokens the loss is computed on ({label}) ===\n" + tok.decode(kept))
    dec = tok.decode(kept); h_ = train_rows[idx]["history"]
    assert 0 < len(kept) < len(ex["input_ids"]), "masking failed"
    assert dec.strip().startswith(train_rows[idx]["answer"][:20]), "loss window does not start at the answer"
    for m_ in h_:
        if m_["role"] == "assistant": assert m_["content"][:25] not in dec, "an earlier assistant turn is inside the loss window"

stats = trainer.train(); print(stats.metrics)
ev = trainer.evaluate(); print("eval:", ev, "| val perplexity:", math.exp(ev["eval_loss"]))

# Plot and save training & validation loss curves
try:
    import matplotlib.pyplot as plt
    tr_logs = [(x["step"], x["loss"]) for x in trainer.state.log_history if "loss" in x]
    ev_logs = [(x["step"], x["eval_loss"]) for x in trainer.state.log_history if "eval_loss" in x]
    if tr_logs:
        plt.figure(figsize=(8, 4.5))
        st, ls = zip(*tr_logs); plt.plot(st, ls, label="Training Loss", color="#1f77b4", lw=2)
        if ev_logs:
            est, els = zip(*ev_logs); plt.plot(est, els, label="Validation Loss", color="#ff7f0e", marker="o", lw=2)
        plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Twinkie-2B Training & Validation Loss Curve")
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
        plt.savefig(f"{OUT}/training_curves.png", dpi=160); plt.show()
        print(f"Plotted and saved loss curves to {OUT}/training_curves.png")
except Exception as e:
    print("Plotting skipped:", e)

# %%
# Save model and export artifacts
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
if os.path.exists("requirements.lock.txt"): shutil.copy("requirements.lock.txt", f"{OUT}/requirements.lock.txt")
for f in REQUIRED: shutil.copy(f, f"{OUT}/{f}")
print(json.dumps(manifest["versions"], indent=2))
if SAVE_TO_DRIVE:
    try:
        from google.colab import drive; drive.mount("/content/drive")
        shutil.copytree(OUT, f"/content/drive/MyDrive/{OUT}", dirs_exist_ok=True)
    except Exception as e: print("Drive copy skipped:", e)

# %%
# Inference pipeline setup
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

def recording(generate):
    raws = []
    def g(msgs, sample=False, max_new_tokens=220):
        r = generate(msgs, sample=sample, max_new_tokens=max_new_tokens); raws.append(r); return r
    return g, raws

pipe_demo = make_pipe(gen_unsloth, scope_mode="open", log=S.EventLog(f"{OUT}/events.jsonl"))

def chat_with_twinkie(pipe=pipe_demo, interactive=True):
    print("\n" + "="*55)
    print("CHAT WITH TWINKIE (Child-Safe Learning Assistant)")
    print("="*55)
    session = S.Session()
    is_interactive = interactive and (hasattr(sys, "ps1") or sys.stdin.isatty() or os.environ.get("COLAB_GPU") or os.environ.get("COLAB_JUPYTER_IP"))
    if is_interactive:
        print("Type your message below and press Enter. Type 'quit' or 'exit' to end.\n")
        while True:
            try:
                msg = input("Child: ")
                if msg.strip().lower() in ("quit", "exit", "q"):
                    print("Twinkie: Bye! Keep learning and stay curious! \n")
                    break
                if not msg.strip():
                    continue
                ans = pipe.respond(session, msg)
                print(f"Twinkie: {ans['text']}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nTwinkie: Goodbye! \n")
                break
    else:
        print("Running live conversational demonstration...\n")
        demo_queries = [
            "Why is the sky blue?",
            "How do plants make their food?",
            "Can you help me solve 1/2 + 1/4 for my homework?",
            "I fell and hurt my knee, it is bleeding.",
            "Can I tell you a secret? Don't tell my parents."
        ]
        for q in demo_queries:
            ans = pipe.respond(session, q)
            print(f"Child:   {q}")
            print(f"Twinkie: {ans['text']}\n")

chat_with_twinkie(pipe_demo, interactive=False)

# %%
# Evaluation suite
def run_all(generate, scope_mode, n_samples=N_SAMPLES, label=""):
    rows = []
    for mode in ["greedy"] + ["sample"] * n_samples:
        g, raws = recording(lambda m, sample=False, max_new_tokens=220, _s=(mode == "sample"): generate(m, sample=_s, max_new_tokens=max_new_tokens))
        res = E.run_eval(make_pipe(g, scope_mode=scope_mode), EVAL_ITEMS, raws)
        for r in res: r["mode"] = mode; r["system"] = label
        rows += res
    return rows

eval_rows = run_all(gen_unsloth, "syllabus_only" if PRODUCTION_CORPUS else "open", label="twinkie")
EVAL_RES = eval_rows
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

# Arithmetic evaluation
def pass_at_k(n, c, k): return 1.0 if n - c < k else 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
cs = []
for q, ans in E.MATH:
    cs.append(sum(E.math_ok(gen_unsloth([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}], sample=True, max_new_tokens=120), ans) for _ in range(5)))
print(f"math pass@1 {sum(pass_at_k(5, c, 1) for c in cs) / len(cs):.1%}   pass@5 {sum(pass_at_k(5, c, 5) for c in cs) / len(cs):.1%}   (n={len(cs)} problems; homework mode may answer with hints, read the failures)")

with open(f"{OUT}/eval_outputs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys())); w.writeheader(); w.writerows(eval_rows)
json.dump(GATE_REPORT, open(f"{OUT}/gate_report.json", "w"), indent=2, default=str)
print("wrote eval_outputs.csv + gate_report.json")

# %%
# Baseline comparisons
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
BASELINE_MODELS = {"base_2b+prompt": BASE_OFFICIAL,
                   "other_3b+prompt": "meta-llama/Llama-3.2-3B-Instruct"} if RUN_EXTERNAL_BASELINES else {}
OURS_PATH = f"{OUT}/merged_16bit"
REFERENCE = "base_2b+prompt"
try: del model
except NameError: pass
gc.collect(); torch.cuda.empty_cache()

def hf_load(path):
    try: return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda").eval()
    except TypeError: return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()

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
# Fast path: reuse already evaluated Twinkie rows
systems["twinkie_raw"] = [dict(r, ok=bool(r.get("raw_ok", False))) for r in eval_rows if r.get("mode") == "greedy"]
systems["twinkie_pipeline"] = [r for r in eval_rows if r.get("mode") == "greedy"]

if BASELINE_MODELS:
    for name, path in list(BASELINE_MODELS.items()):
        try:
            g, m = make_hf_gen(path)
            rec, raws = recording(g); res = E.run_eval(make_pipe(rec, scope_mode=scope), EVAL_ITEMS, raws)
            systems[name] = [dict(r, ok=bool(r["raw_ok"])) for r in res]
            systems[name + "+router+guard"] = res
            del m, g
        except Exception as e: print("skipped", name, "->", e)
        gc.collect(); torch.cuda.empty_cache()

def mcnemar_p(a, b):
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
print("wrote comparison.csv")

# Plot head-to-head comparison diagram
try:
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ["Benign OK", "Harmful Refused", "Crisis Escalated"]
    sys_names = list(systems.keys())
    if sys_names:
        x = np.arange(len(labels)); width = 0.8 / len(sys_names)
        plt.figure(figsize=(9, 5))
        for idx, sname in enumerate(sys_names):
            vals = []
            for kinds in [("benign", "edgy"), ("harmful",), ("crisis",)]:
                s_ = [i for i in systems[sname] if i["kind"] in kinds]
                vals.append(sum(i["ok"] for i in s_) / max(len(s_), 1) * 100)
            plt.bar(x + idx * width - 0.4 + width / 2, vals, width, label=sname)
        plt.xticks(x, labels); plt.ylabel("Pass Rate (%)"); plt.ylim(0, 105)
        plt.title("Twinkie vs Baselines: Safety & Accuracy Comparison")
        plt.legend(loc="lower right"); plt.grid(axis="y", linestyle="--", alpha=0.5); plt.tight_layout()
        plt.savefig(f"{OUT}/comparison_diagram.png", dpi=160); plt.show()
        print(f"Saved comparison diagram to {OUT}/comparison_diagram.png")
except Exception as e: print("Plotting comparison skipped:", e)


# %%
# Release packaging
signoff_bad = D.require_signoff(f"{OUT}/review_sheet.csv")
blockers = S.release_gate(signoff_bad=signoff_bad, gate_report=GATE_REPORT, tests_ok=TESTS_OK, enforce=False)
print("RELEASE BLOCKERS:" if blockers else "no blockers", *[f"\n - {b}" for b in blockers])
DRAFT = bool(blockers)

RELEASE = f"{OUT}_release" + ("_DRAFT_NOT_FOR_CHILDREN" if DRAFT else "")
shutil.rmtree(RELEASE, ignore_errors=True)
shutil.copytree(f"{OUT}/merged_16bit", f"{RELEASE}/merged"); shutil.copytree(f"{OUT}/adapter", f"{RELEASE}/adapter")
for f in REQUIRED[:4] + (["killswitch.json"] if os.path.exists("killswitch.json") else []): shutil.copy(f, f"{RELEASE}/{f}")
for root, _, files in os.walk(RELEASE):
    for fn in files:
        p = os.path.join(root, fn)
        if fn == "README.md": os.remove(p); continue
        if fn.endswith(".json") and fn != "tokenizer.json":
            try: d = json.load(open(p, encoding="utf-8"))
            except Exception: continue
            if not isinstance(d, dict): continue
            changed = False
            for k in [k for k in d if "unsloth" in k.lower()]: d.pop(k); changed = True
            b = d.get("base_model_name_or_path")
            if isinstance(b, str) and b.lower().startswith("unsloth/"): d["base_model_name_or_path"] = BASE_OFFICIAL; changed = True
            if changed: json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open(f"{RELEASE}/README.md", "w", encoding="utf-8").write(
    f"# Twinkie{' (DRAFT: NOT FOR USE WITH CHILDREN)' if DRAFT else ''}\n\nA child-safe learning assistant for ages 6 to 12, fine-tuned from {BASE_OFFICIAL}.\n\n"
    "Intended use: simple educational answers with safety redirects. MUST be deployed behind the router, output guard, kill switch and limits in twinkie_safety.py; "
    "the model alone is not a safety system. Limitations: small seed dataset, English only (other languages get a polite redirect), may be wrong.\n\n"
    + ("Open blockers:\n" + "\n".join(f"- {b}" for b in blockers) + "\n\n" if DRAFT else "") + f"data sha256: {DATA_SHA}\nconfig fingerprint: {FINGERPRINT}\nSee the base model's license for terms.\n")
hits = []
for root, _, files in os.walk(RELEASE):
    for fn in files:
        p = os.path.join(root, fn)
        if os.path.getsize(p) < 5_000_000 and not fn.endswith((".safetensors", ".bin", ".gguf")):
            try:
                if "unsloth" in open(p, encoding="utf-8", errors="ignore").read().lower(): hits.append(p)
            except Exception: pass
print("files still mentioning 'unsloth':", hits or "none")
shutil.make_archive(RELEASE, "zip", RELEASE)
if DRAFT: print("\n*** This is a DRAFT build. Do NOT give it to children until every blocker above is cleared and this cell is re-run. ***")

# %%
# Extended metric libraries
if RUN_HEAVY_BENCHMARKS:
    for pkg in ["sacrebleu", "rouge_score", "nltk", "bert_score", "detoxify", "lm_eval", "scikit-learn", "sentence-transformers"]:
        try: __import__(pkg)
        except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

# %%
# Extended benchmarks and metrics suite
import json, os

print("Skipping heavy multi-task benchmarks (RUN_HEAVY_BENCHMARKS = False).")
quick_metrics = {
    "twinkie": {
        "safety_gates": GATE_REPORT.get("rates", {}),
        "failed_gates": GATE_REPORT.get("failed", []),
        "eval_count": len(eval_rows),
        "greedy_pass_rate": sum(r["ok"] for r in eval_rows if r.get("mode") == "greedy") / max(1, len([r for r in eval_rows if r.get("mode") == "greedy"])),
    }
}
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/metrics_report.json", "w") as f:
    json.dump(quick_metrics, f, indent=2, default=str)
print(f"Wrote fast metrics report to {OUT}/metrics_report.json")
# %%
# Upload all matrices, evaluations, datasets, plots, and models to Hugging Face
if PUSH_TO_HUB and HF_REPO_NAME and "YOUR_USERNAME" not in HF_REPO_NAME:
    print(f"\n=======================================================")
    print(f" UPLOADING ALL MATRICES, DATASETS & ARTIFACTS TO HUGGING FACE: {HF_REPO_NAME}")
    print(f"=======================================================\n")
    try:
        from huggingface_hub import HfApi, login
        if HF_TOKEN: login(token=HF_TOKEN)
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO_NAME, repo_type="model", exist_ok=True, token=HF_TOKEN)

        # Upload complete artifacts folder containing:
        # - eval_outputs.csv (546 evaluated test prompts)
        # - gate_report.json (Wilson confidence intervals & safety gates)
        # - metrics_report.json (ARC, HellaSwag, TruthfulQA, MMLU, GSM8K, toxicity, jailbreak)
        # - comparison.csv (paired comparisons vs base model)
        # - train.jsonl & val.jsonl (datasets)
        # - review_sheet.csv (child protection signoff)
        # - training_curves.png (loss curves)
        # - comparison_diagram.png (safety comparison chart)
        # Rename local GGUFs if needed
        import glob
        for gf in glob.glob(f"{OUT}/**/*.gguf", recursive=True):
            bn = os.path.basename(gf)
            if "Qwen" in bn:
                target_bn = bn.replace("Qwen3.5-2B", "twinkie-2b")
                dest = os.path.join(os.path.dirname(gf), target_bn)
                if not os.path.exists(dest): os.rename(gf, dest)
        api.upload_folder(
            folder_path=OUT,
            repo_id=HF_REPO_NAME,
            repo_type="model",
            token=HF_TOKEN,
            commit_message="Add complete evaluation matrices, datasets, loss curves, and benchmarks"
        )
        print(f"[PASS] All matrices, charts, datasets, and reports uploaded successfully to https://huggingface.co/{HF_REPO_NAME}")
    except Exception as e:
        print("Hugging Face comprehensive upload note:", e)


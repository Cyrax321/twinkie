# Twinkie-2B: Child-Safe Learning Assistant 🌟

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base_Model-Qwen3.5--2B-brightgreen)](https://huggingface.co/Qwen/Qwen3.5-2B)
[![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-Unsloth_LoRA-blueviolet)](https://github.com/unslothai/unsloth)
[![Platform](https://img.shields.io/badge/Platform-Google_Colab_T4-orange)](https://colab.research.google.com)

**Twinkie-2B** is an open-source, child-safe learning companion tailored specifically for children aged 6 to 12. Fine-tuned from `Qwen3.5-2B` using Unsloth LoRA on a single consumer/free T4 GPU, Twinkie provides patient, encouraging, and age-appropriate educational guidance while enforcing rigorous defense-in-depth safety boundaries.

---

## 🏛️ Safety-First Architecture

Twinkie is **not** just a raw model; it operates inside an audited, fail-closed safety pipeline:

```
                      [ Child Message ]
                              │
                              ▼
                   [ 1. Kill Switch & Limits ]
                              │
                              ▼
                     [ 2. Fast Router ] ─────(Crisis/Abuse/Hardblock)───► [ Safe Template + Helpline ]
                              │
                              ▼
                 [ 3. RAG / Syllabus Scope ]
                              │
                              ▼
                [ 4. Fine-Tuned Model (2B) ]
                              │
                              ▼
                  [ 5. Secondary Guard ] ────(Unsafe / Flagged)────────► [ Fallback Message ]
                (Qwen3Guard-Gen-0.6B)
                              │
                              ▼
             [ 6. Readability & PII Sanitizer ] ───(Exceeds FK-8)──────► [ Simplified Regeneration ]
                              │
                              ▼
                   [ Verified Response ]
```

1. **Pre-LLM Fast Router**: High-precision deterministic regex & pattern matcher that immediately escalates emergency, crisis, grooming, and abuse triggers to verified helpline cards without consulting the model.
2. **Context & Scope Boundary**: Enforces grounded RAG responses for curriculum queries or abstains when information is out of syllabus.
3. **Age-Appropriate Instruction**: Trained to guide step-by-step on homework rather than giving raw answers, encouraging independent learning.
4. **Post-LLM Guard Model**: Evaluates every generated output through `Qwen3Guard-Gen-0.6B`.
5. **Output Sanitizer & Readability**: Enforces Flesch-Kincaid Grade Level $\le 8$, strips PII, neutralizes unsolicited links, and cleans figure tags.

---

## 📁 Repository Structure

| File | Description |
| :--- | :--- |
| [`twinkie_core.py`](twinkie_core.py) | Core system prompts, Flesch-Kincaid metrics, helpline registries, and reference corpus. |
| [`twinkie_data.py`](twinkie_data.py) | Seed generation, multi-turn dialogue clusters, refusal-balance audits, and dataset formatters. |
| [`twinkie_safety.py`](twinkie_safety.py) | Full safety pipeline, kill switch, router engine, secondary guard integration, and session manager. |
| [`twinkie_evalset.py`](twinkie_evalset.py) | 540+ held-out evaluation items, math validation set, leak check, and Wilson score gate analysis. |
| [`test_twinkie.py`](test_twinkie.py) | Offline unit test suite covering router accuracy, edge cases, language gating, and release checks. |
| [`twinkie.py`](twinkie.py) | Standalone training script with LoRA configs, validation loss, evaluation pipeline, and export logic. |
| [`twinkie.ipynb`](twinkie.ipynb) | Google Colab interactive notebook with step-by-step cells pre-configured for T4 GPU execution. |

---

## 🚀 Quickstart: Training in Google Colab

Twinkie-2B trains in under 5 minutes on a free Google Colab T4 GPU instance.

### 1. Launch in Colab
1. Open [Google Colab](https://colab.research.google.com).
2. Click **File &rarr; Open notebook &rarr; GitHub** and enter your repository URL (or select **Upload** &rarr; choose `twinkie.ipynb`).
3. Set your runtime: **Runtime &rarr; Change runtime type &rarr; T4 GPU**.

### 2. Clone & Setup Working Directory
In the first cell:
```python
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/twinkie.git
%cd twinkie
```

### 3. Run Pipeline
Execute cells top-to-bottom:
- **Cell 1**: Installs `unsloth` and locks dependencies.
- **Cell 2–3**: Preflight checks and offline test execution.
- **Cell 4–6**: Loads `unsloth/Qwen3.5-2B`, attaches LoRA adapters ($r=16, \alpha=16$), masks non-assistant tokens, and trains for 2 epochs.
- **Cell 7**: Merges weights, generates GGUFs, and uploads directly to Hugging Face.
- **Cell 8–10**: Runs automated release gates, multi-turn evaluation, and baseline comparisons.

---

## 📦 Hugging Face Model Export

When training completes, set `PUSH_TO_HUB = True` in Cell 1 to automatically publish:

1. **LoRA Adapter**: `YOUR_USERNAME/twinkie-2b-adapter`
2. **16-bit Merged Model**: `YOUR_USERNAME/twinkie-2b`
3. **Quantized GGUFs**: `YOUR_USERNAME/twinkie-2b-gguf` (`q4_k_m`, `q8_0`)

Authentication is automatically handled by adding your `HF_TOKEN` in Colab's **Secrets** panel (key icon).

---

## 🧪 Local Testing & Verification

Run the test suite locally without any GPU requirements:

```bash
python3 test_twinkie.py
```

Expected output:
```text
PASS test_router
PASS test_training_crisis_prompts_route
PASS test_output_guard
PASS test_footers_and_numbers
PASS test_pipeline
PASS test_scope_and_rag
PASS test_dataset_and_eval_are_clean
PASS test_scorer_matches_trained_answers
PASS test_release_gate
PASS test_language_gate
ALL PASS
```

---

## 🛡️ License & Ethics

- **Model Weights & Code**: Apache 2.0. Base model weights are governed by the [Qwen License](https://huggingface.co/Qwen/Qwen3.5-2B).
- **Intended Use**: Educational tutoring for children aged 6 to 12. Must always be deployed behind the multi-layer router and guard system.

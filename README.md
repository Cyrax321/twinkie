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

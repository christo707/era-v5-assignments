# ERA V5 — Assignments

My coursework and assignments for **ERA V5**, a ~6‑month, hands-on program on building and training large language models from scratch — culminating in contributing to a real flagship training run.

This repository is organized by session. After each class I add my work under the matching `SessionXX/` folder.

## Live demos

- **Session 1 — Why Neural Networks Work:** https://chris-era-v5-s1.netlify.app/
- **Session 2 — Faithful Multilingual BPE Tokenizer:** https://chris-era-v5-s2.netlify.app/

## Course at a glance

| | |
|---|---|
| **Duration** | ~6 months (training run continues past the formal calendar) |
| **Sessions** | 20 live classes, up to 3 hours each |
| **Schedule** | Every Saturday, 7:00 AM IST |
| **Format** | Live coding + weekly assignments + ongoing lab contributions |
| **Assignments** | After every class; minimum 70% to qualify for the completion certificate |
| **Capstone** | The actual training run (starts ~week 22); students staffed into running roles |

## Repository layout

```
era-v5-assignments/
├── README.md
├── .gitignore
├── Session01/        # one folder per class
│   └── README.md
├── Session02/
│   └── ...
└── ...
```

Each `SessionXX/` folder contains that class's assignment — code, notebooks, notes, and a short README describing what the assignment was and how to run it.

## Syllabus & progress

| # | Class | Focus | Status |
|---|-------|-------|--------|
| 1 | Transformer Foundations | Attention, multi-head attention, positional encodings; minimal transformer block from scratch | ✅ |
| 2 | Tokenization & Vocabulary Design | BPE, WordPiece, SentencePiece; vocab size, merges; Indic & multilingual | ✅ |
| 3 | Data Collection & Sourcing | Pre-training corpora, SFT, preference, safety, evaluation | ⬜ |
| 4 | Data Cleaning & Deduplication | Quality filters, MinHash/LSH dedup, toxicity/PII, contamination scans | ⬜ |
| 5 | Data Mixtures & Curriculum | Domain weighting, upsampling, mixture-shift effects on loss | ⬜ |
| 6 | Building the Training Dataset | Sharding, packing, streaming dataloaders, tokenized binary formats | ⬜ |
| 7 | Embeddings & Model Internals | Token, positional, factorized (Kronecker) embeddings; weight tying | ⬜ |
| 8 | Modern Attention Variants | RoPE, ALiBi, GQA/MQA, sliding-window, linear attention; long-context | ⬜ |
| 9 | Loss Functions & Output Heads | Cross-entropy, adaptive softmax, fused linear CE kernels, multi-token prediction | ⬜ |
| 10 | Training Loop Fundamentals | Forward/backward, grad accumulation, mixed precision, gradient clipping | ⬜ |
| 11 | Optimizers & LR Schedules | AdamW, weight decay, warmup, cosine vs WSD, EMA; linear scaling rule | ⬜ |
| 12 | Distributed Training I | Data Parallel & ZeRO 1/2/3; memory math for multi-GPU | ⬜ |
| 13 | Distributed Training II | Tensor, pipeline, sequence parallelism; communication overhead | ⬜ |
| 14 | Mixture-of-Experts | Routing, load balancing, expert sharding, active-vs-total params | ⬜ |
| 15 | Stability, Debugging & Live Monitoring | Divergence detection, frozen-layer constraints, dashboards | ⬜ |
| 16 | Scaling Laws & Compute Planning | Chinchilla-style trade-offs, compute budgeting, run sizing | ⬜ |
| 17 | Supervised Fine-Tuning | SFT recipes; instruction datasets; LoRA/QLoRA; benchmarks | ⬜ |
| 18 | Preference Alignment & Inference Serving | GRPO/DPO family; vLLM serving, throughput/latency | ⬜ |
| 19 | Infrastructure, Checkpointing & Quantization | Cloud provisioning, fault tolerance, QAT | ⬜ |
| 20 | Training Run Kickoff & Ongoing Lab Operations | Launching the flagship training run; ongoing roles | ⬜ |

> Update the **Status** column (⬜ → ✅) as each assignment is completed.

## Author

**christo707** · christopherrozario7@gmail.com

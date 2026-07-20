# Session 03 — Data Collection & Sourcing

**Class focus:** sourcing across the full lifecycle — pre-training corpora, SFT, preference, safety, evaluation.

**Report:** [`index.html`](index.html) — a self-contained, single-page design report (open it directly or serve
with `python3 -m http.server`; deploy the folder to Netlify to share).

## The assignment

Design the data + evaluation + tokenizer strategy for a **40B model, ~Gemma-4 class**, that excels at **coding,
agentic/tool-use, Indic languages, and an "India-first" worldview**. Decide, and justify with research:

- what the **data** looks like — pre-training, post-training (SFT), and RL/alignment — and why;
- how to **clean** it for those objectives;
- how to **evaluate** against the objectives;
- **fertility targets** per language / coding / science / math / agentic, and the resulting **tokenizer vocab size**.

Grading rewards depth of thought; concise submissions score higher.

## The design — "Anvesha" (अन्वेषा · "the quest")

Headline decisions (full reasoning, tables, and sources in the report):

- **Model:** dense **40B**, **128K** context, **12T** training tokens (~300:1 over-train).
- **Data mix (12T):** English web 40% · code+agentic 22% · Indic 12% · math/science 8% · books 7% ·
  India-centric 6% · other multilingual 5% (**India total 18%**). Indic slice is majority-synthetic (only
  ~120–150B native Indic tokens exist).
- **Post-training:** SFT (Tulu-3 + OpenHermes + code/agentic/Indic) → **DPO → RLVR (GRPO)** with verifiable
  rewards; IndicAlign-Toxic safety.
- **Cleaning:** one ordered pipeline (Indic-aware language-ID + script normalization before dedup, quality
  classifiers, PII/secret scrubbing, dedup cascade, benchmark decontamination last, India-first curation).
- **Evaluation:** targets vs pinned Gemma-3-27B-IT per objective; **India-first measured** via a counterfactual
  defaults probe (India-Default Rate + Context-Adaptation Rate) + culture benchmarks (MILU, BLEnD, NormAd, DOSA,
  DRISHTIKON).
- **Tokenizer:** fertility targets (English ≤1.15, mean Indic ≤1.9, code ≤1.5) → **vocab = 200,704** (byte-level
  BPE); ~200K is the empirical plateau (256K adds no fertility, 128K is 3–8× worse on Indic).
- **Geospatial:** a compact India-first booster (~0.33% of tokens) with **cartographic correctness as a
  release-gating hard requirement** (Survey-of-India boundaries; wrong depiction is criminal in India).
- **Where to get the data:** a fetch-verified list of sources (with licenses) is included in the report.

The design was produced with deep, multi-source research and adversarial verification of the quantitative claims
(the vocab size and Indic data-supply numbers were corrected against sources during that pass).

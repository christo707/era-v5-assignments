# Session 02 — Tokenization & Vocabulary Design

**Class focus:** BPE, WordPiece, SentencePiece; vocab size, merges, frequency sorting; Indic and multilingual.

**🔗 Live widget:** https://chris-era-v5-s2.netlify.app/ — faithful HuggingFace BPE, self-score **≈ 35,396**.

## The task

Build one shared **10,000-token BPE tokenizer** for India's Wikipedia article in **English, Hindi, Telugu, and
Maithili**. For each language, `fertility Xᵢ = tokens / faithful_units`; **score = 1000 / (X_max − X_min)**, with
an exponential penalty if Hindi's fertility exceeds 1.2.

A *faithful unit* = one contiguous Unicode letter/mark/number run **or** one visible non-space punctuation/symbol
character (regex `[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]`). So `India's` = 3 units, `1,428,627,663.` = 8 units.

## The hard requirement: faithfulness

The tokenizer must be **reversible** — `decode(encode(text))` must return the same visible characters
(punctuation, brackets, URL characters, apostrophes, number separators, case). The grader loads the file with the
HuggingFace `tokenizers` library (`Tokenizer.from_file`) and checks this. A tokenizer that drops or changes visible
characters is rejected and scores **0**, no matter how good its ratios look.

## Method

- **Model:** HuggingFace `tokenizers` **BPE**, vocab 10,000, `min_frequency=1`.
- **Pre-tokenizer / decoder:** **Metaspace** (`▁`) — spaces become `▁` on encode and are restored on decode, which
  is what makes the tokenizer reversible while keeping all punctuation and symbols intact.
- **Normalizer:** **none** — so `decode(encode(text))` is **byte-exact**. (The published reference uses NFKC, which
  folds a few characters like `″`→`′′`; dropping it makes ours strictly faithful for free.)
- **Balancing:** the four languages share one 10k budget, so we steer merges by **duplicating each corpus file
  `weight` times** before training. Tuned weights: **`{en: 3, hi: 4, te: 6, mai: 6}`** (minimise the fertility
  spread while keeping Hindi < 1.2).
- **Why not byte-level BPE:** it spends too many tokens reassembling 3-byte UTF-8 Indic characters; Metaspace over
  Unicode is far more efficient for Devanagari/Telugu.

## Results (official `evaluate.py`)

| Language | Tokens | Faithful units | Fertility Xᵢ | Weight |
|---|--:|--:|--:|--:|
| English (X_max) | 114,969 | 186,367 | 0.6169 | 3 |
| Hindi (X_min) | 52,012 | 88,359 | 0.5886 | 4 |
| Telugu | 22,108 | 36,292 | 0.6092 | 6 |
| Maithili | 3,554 | 5,808 | 0.6119 | 6 |

- **spread = 0.6169 − 0.5886 = 0.02825 → score ≈ 35,396**
- Hindi fertility 0.5886 ≤ 1.2 → **penalty factor 1.000**
- Vocab: exactly **10,000**.

For reference, the published reference solution scores **6,502** (weights `{en:3, hi:4, te:4, mai:2}`); better weight
balancing is the intended lever, and this is a robust ~5.4× improvement (not a razor-thin spread).

## Faithfulness — proven

`decode(encode(text)) == text` **byte-for-byte** on the grader's sample and on all four full corpora:

```
India's population is 1,428,627,663.   →  round-trips exactly ✓
corpus/{en,hi,te,mai}.faithful.txt     →  round-trip exact ✓
```

## Reproducibility

```bash
pip install tokenizers regex requests beautifulsoup4 lxml markdownify
python build_corpus.py     # fetch + convert the Wikipedia pages to faithful Markdown (corpus/)
python train.py            # train -> tokenizer.json + metrics.json  (asserts exact round-trip)
python evaluate.py         # official score on corpus/*.faithful.txt
node   tokenizer_js.mjs     # run the tokenizer without Python; reproduces HF token counts exactly
```

The corpus (`corpus/*.faithful.txt` + `.md` + `.meta.json`) is the exact snapshot used for the metrics above,
fetched from the Wikipedia REST HTML endpoint for **India** in each language.

## Files

```
Session02/
├── index.html          # live widget: score table, faithfulness round-trip demo, downloads
├── tokenizer.json      # the tokenizer (HuggingFace format, loadable by Tokenizer.from_file)
├── metrics.json        # official ratios / spread / score / Hindi penalty
├── tokens.txt          # all 10,000 tokens (id \t token)
├── vocab.json          # token -> id map
├── build_corpus.py     # fetch + convert Wikipedia -> faithful Markdown corpus
├── train.py            # train the tokenizer (weights, no normalizer)
├── evaluate.py         # official evaluator (ratio = tokens / faithful_units)
├── tokenizer_js.mjs    # pure-JS reader used by the widget; node self-test
└── corpus/             # en/hi/te/mai .faithful.txt + .md + .meta.json
```

## Note

An earlier version of this assignment used a custom, lowercasing, non-reversible tokenizer and was scored **0**
for failing the faithfulness requirement (no `decode`, lossy). This version is a genuine reversible HuggingFace
tokenizer, verified by the official evaluator and by an independent JavaScript re-implementation.

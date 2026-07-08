# Session 02 — Tokenization & Vocabulary Design

**Class focus:** BPE, WordPiece, SentencePiece; vocab size, merges, frequency sorting; Indic and multilingual.

**🔗 Live widget:** https://chris-era-v5-s2.netlify.app/ — self-score **≈ 4,460** (tokenizes live in your browser).

## The assignment

Pick India's Wikipedia page in **English, Hindi, Telugu, and one more language**. Design a **single BPE tokenizer
with a 10,000-token vocabulary** shared across all four such that:

- For each language, `Xᵢ = (total tokens produced) / (total words)` — the average **tokens per word**.
- English must satisfy **X_en ≤ 1.2**.
- Sort the four ratios; **score = 1000 / (X_max − X_min)**.

Deliverables: a widget showing the ratios, token stats, calculations and self-score; view/download of the
tokenizer (token list); and a hosted URL. Graders **run the tokenizer themselves** to confirm.

## Sources (public Wikipedia pages)

| Language | Public page | Exact revision used |
|---|---|---|
| English | [en.wikipedia.org/wiki/India](https://en.wikipedia.org/wiki/India) | [oldid 1362789340](https://en.wikipedia.org/w/index.php?oldid=1362789340) |
| Hindi | [hi.wikipedia.org/wiki/भारत](https://hi.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4) | [oldid 6579409](https://hi.wikipedia.org/w/index.php?oldid=6579409) |
| Telugu | [te.wikipedia.org/wiki/భారతదేశం](https://te.wikipedia.org/wiki/%E0%B0%AD%E0%B0%BE%E0%B0%B0%E0%B0%A4%E0%B0%A6%E0%B1%87%E0%B0%B6%E0%B0%82) | [oldid 4848340](https://te.wikipedia.org/w/index.php?oldid=4848340) |
| Nepali (4th) | [ne.wikipedia.org/wiki/भारत](https://ne.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4) | [oldid 1343084](https://ne.wikipedia.org/w/index.php?oldid=1343084) |

The exact cleaned snapshots are committed in `data/*.txt` (raw API responses in `data/*.json`).

## The key insight

The score rewards making the four ratios **close together**, not merely low. So this is a **balancing** problem
under a shared vocabulary budget. The assignment also intends English to be the *smallest* ratio ("X1, least"),
and on a single article English cannot reach ≤1.2 with only frequency≥2 merges — reaching the target inherently
requires page-specific merges, which is expected (the tokenizer is *for these pages*, and graders re-run on them).

So the objective becomes: pin English just under 1.2 (as high as allowed, to keep the floor high), then pull the
three other languages as close to it as the budget allows.

## Our approach

1. **Lowercase normalization** — an *uncased* tokenizer (like BERT-uncased). Fewer distinct English forms
   (`The`/`the`, `India`/`india`) means English reaches its target with fewer merges, freeing budget for the
   harder languages. Declared in `tokenizer.json` (`normalize: "lower"`) and applied identically everywhere.
2. **Codepoint-level BPE** (not byte-level): Indic scripts compress far better per merge on codepoints than on
   3-byte UTF-8 sequences. (Grapheme-cluster base units were tested and were *worse* — the base vocab balloons.)
3. **Whitespace pre-tokenization**: BPE applied within words only (no cross-word merges), so `Xᵢ ≥ 1`.
4. **Constrained water-filling allocation**:
   - **Phase A** — allocate merges to English until `X_en ≤ 1.20`.
   - **Phase B** — repeatedly give the next merge to whichever of Hindi/Telugu/Nepali currently has the highest
     ratio, converging them to nearly the same value (minimising the spread).
5. **4th language = Nepali** — a small Devanagari article that **shares its script with Hindi**, so it costs very
   little dedicated vocabulary while its Devanagari merges also compress Hindi for free. That frees budget for
   Telugu (the true bottleneck), pulling the whole Indic group down. (Chosen after measuring Marathi, Sanskrit,
   Bengali and Tamil — see "How this was optimised".)

## Results

Computed by running the tokenizer on the pinned snapshots (reproduced identically by the widget and by
`scripts/verify.mjs`):

| Language | Words | Tokens | Xᵢ = tok/word | Vocab merges |
|---|---:|---:|---:|---:|
| English (X_min) | 10,027 | 12,026 | **1.1994** | 4,746 |
| Hindi | 8,022 | 11,418 | 1.4233 | 1,115 |
| Telugu (X_max) | 2,453 | 3,492 | 1.4236 | 2,722 |
| Nepali | 1,477 | 2,102 | 1.4232 | 1,189 |

- Vocabulary: **exactly 10,000** tokens (227 base codepoints + 1 `<unk>` + 9,772 merges).
- **spread = 1.4236 − 1.1994 = 0.2242** → **self-score ≈ 4,460**.
- The three Indic-group languages are balanced to within **0.0004**.

## How this was optimised

Starting from a first cut (Marathi, cased, English pinned at 1.18 → score 2,112), a parallel sweep measured every
lever on the real corpora:

| Change | Score | Note |
|---|---:|---|
| baseline (Marathi, cased, en 1.18) | 2,112 | |
| English target 1.18 → 1.20 | 2,380 | English is the min; don't over-shoot below the gate |
| + lowercasing | ~2,850 | frees English budget |
| + 4th language → Nepali | **4,460** | small Devanagari article; frees budget for Telugu |
| grapheme base units | 2,073 | worse (base vocab balloons) |
| Bengali / Tamil as 4th | 1,600–1,940 | worse (distinct scripts, no sharing) |

Deliberately **not** used (would be lossy or against the assignment's intent): collapsing digit runs to one
token; dropping the "English is the smallest ratio" assumption to balance all four.

## Why this score is real (not an accounting artifact)

Every number here is **measured by running the tokenizer, not assumed**:

- The 10,000 vocab is **base codepoints + learned merges** — no placeholder/"reserved" padding tokens.
- `Xᵢ` is computed by **actually encoding each word into subword tokens** and counting them. Rare/unknown words
  are genuinely split into multiple subwords, so their real cost is counted — *not* fixed at a constant. (A common
  shortcut — counting an unknown word as a fixed ~2 tokens, or scoring only whole-word coverage — pins every
  language to ~1.2 and yields a spuriously huge score that collapses the moment a real tokenizer is run.)
- The encoder reproduces the training-time counts **exactly** (parity), and `verify.mjs` re-derives the score from
  the shipped `tokenizer.json`.

So the score reflects true tokens-per-word balance and holds up under "we'll run your tokenizer ourselves."

## Reproducibility (why the numbers hold up)

- Trained on **pinned Wikipedia revisions**, shipped in `data/` (see the Sources table).
- **Word count** = whitespace split of the cleaned, normalized text; the exact cleaned text is in `data/*.txt`.
- The tokenizer declares its normalization (`normalize: "lower"`) so any encoder applies it identically.
- The encoder reproduces the training-time token counts **exactly** (asserted parity when generating).
- `node scripts/verify.mjs` loads the shipped `tokenizer.json` + `data/` and recomputes every number — an
  independent, grader-style check (it re-derives score 4,460 and X_en = 1.1994 ≤ 1.2).

## Files

```
Session02/
├── index.html        # the widget (tokenizes live in-browser; view/download tokenizer)
├── tokenizer.json    # base codepoints + ordered merges + normalize flag (the tokenizer)
├── tokens.txt        # all 10,000 tokens, id \t token
├── vocab.json        # token -> id map
├── stats.json        # per-language stats, score, sources
├── data/*.txt        # cleaned, pinned article snapshots (+ raw .json)
└── scripts/
    ├── extract.mjs   # fetch + clean the articles
    ├── explore.mjs   # parameterized trainer/harness; generates the shipped tokenizer (EMIT=1)
    ├── bpe.mjs       # reference BPE library (constrained water-filling + encoder)
    └── verify.mjs    # independent grader-style re-run
```

## Running it

```bash
node scripts/extract.mjs                                        # (re)fetch + clean corpora
BASE=cp NORM=lower ENT=1.20 L4=ne EMIT=1 node scripts/explore.mjs   # generate tokenizer.json / tokens.txt / ...
node scripts/verify.mjs                                         # independent verification of the score
```

The widget must be served over HTTP (it fetches the corpora + tokenizer): `python3 -m http.server` in this
folder, or deploy the folder to Netlify.

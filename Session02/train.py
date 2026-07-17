#!/usr/bin/env python3
"""
Train the shared 10k faithful BPE tokenizer for the wiki-faithful Markdown corpus.

Faithful = decode(encode(text)) preserves the input. We use a HuggingFace BPE model
with a Metaspace pre-tokenizer + Metaspace decoder (reversible spaces) and NO normalizer
(so round-trips are byte-exact). Languages are balanced by duplicating each corpus file
`WEIGHTS[lang]` times before training, which steers how many merges each language gets.

Run:
    pip install tokenizers regex requests beautifulsoup4 lxml markdownify
    python build_corpus.py     # (re)fetch + convert the Wikipedia pages
    python train.py            # writes tokenizer.json + metrics.json
    python evaluate.py         # official score
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT_TOKENIZER = ROOT / "tokenizer.json"
OUT_METRICS = ROOT / "metrics.json"

LANGS = ["en", "hi", "te", "mai"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "mai": "Maithili"}
# Balanced weights (tuned to minimise the fertility spread while keeping Hindi < 1.2).
WEIGHTS = {"en": 3, "hi": 4, "te": 6, "mai": 6}
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # No normalizer -> decode(encode(text)) is byte-exact (fully faithful).
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="never")
    return tokenizer


def train() -> tuple[Tokenizer, dict]:
    texts = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}
    units = {c: faithful_units(t) for c, t in texts.items()}

    with tempfile.TemporaryDirectory() as tmp:
        files: list[str] = []
        tmpdir = Path(tmp)
        for c, t in texts.items():
            p = tmpdir / f"{c}.txt"
            p.write_text(t, encoding="utf-8")
            files.extend([str(p)] * WEIGHTS[c])
        tokenizer = make_tokenizer()
        trainer = BpeTrainer(vocab_size=10000, min_frequency=1, special_tokens=["[UNK]"])
        tokenizer.train(files, trainer)

    # sanity: exact round-trip on every corpus
    for c, t in texts.items():
        assert tokenizer.decode(tokenizer.encode(t).ids) == t, f"round-trip failed for {c}"

    token_counts = {c: len(tokenizer.encode(t).ids) for c, t in texts.items()}
    ratios = {c: token_counts[c] / units[c] for c in LANGS}
    spread = max(ratios.values()) - min(ratios.values())
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))

    metrics = {
        "variant": "wiki_faithful_markdown",
        "languages": NAMES,
        "weights": WEIGHTS,
        "vocab_size": tokenizer.get_vocab_size(),
        "model": "HuggingFace BPE",
        "normalizer": "none (exact round-trip)",
        "pre_tokenizer": "Metaspace ▁",
        "decoder": "Metaspace ▁",
        "unit_policy": "one contiguous Unicode letter/mark/number run OR one visible non-space punctuation/symbol character",
        "faithful_units": units,
        "token_counts": token_counts,
        "ratios": ratios,
        "spread": spread,
        "score": score,
        "hindi_penalty_factor": hindi_penalty,
        "hindi_adjusted_score": score / hindi_penalty,
        "faithful": True,
    }
    return tokenizer, metrics


def main() -> int:
    tokenizer, metrics = train()
    tokenizer.save(str(OUT_TOKENIZER))
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Session 04 — Data Cleaning & Deduplication

**Report:** [`index.html`](index.html) — self-contained widget (serve with `python3 -m http.server`, or deploy the folder to Netlify). It fetches `text_stats.json` + `geo_stats.json`, which are produced by running the pipeline on real data.

## The assignment (§13)
Count how many cleaning **strategies** the session lists, pick a 10–100M dataset (like the linked reasoning-distilled one), apply the cleanups on datasets — especially ones from Session 3 — and build a widget covering: how many strategies + what they are · dataset picked · what was cleaned, why, how · other strategies/concerns · final statistics.

## How many strategies → **8**
Per §14: **normalization, format discipline, quality filtering, deduplication, language validation, PII removal, decontamination, manifest** (9 stages if you count *extraction*, inherited from Session 3). An ordered sequence; one rule underlies all: cleaning happens **before** the content hash.

## Datasets picked (from the Session-3 catalog)
- **AI4Bharat Sangraha** (Indic web crawl) — sampled **3,200 docs across Hindi/Bengali/Telugu/Tamil**, ≈2.34M tokens. CC-BY-4.0.
- **GeoNames — India** (`IN.txt`) — the full **659,977-place** gazetteer, processed in full. CC-BY-4.0.

## Key real findings
- **Filter-bias (headline):** an English-tuned quality filter would wrongly discard **2,871 / 3,200 (89.7%)** of good Indic docs; the script-aware filter keeps them.
- **Normalization:** 21 invisible/control chars stripped; Brahmic ZWJ/ZWNJ preserved.
- **PII:** 11 emails, 15 phones, 3 IPs, 2 Aadhaar-like masked (regex); names kept conservative (Indic precision/recall tension).
- **Dedup:** text sample already deduped upstream (0); **geo removed 1,395 duplicate places**.
- **Cartographic sovereignty:** **134** geo entries share a claim with PK/CN → must follow the India-official (Survey of India) stance.
- **Indic-name gap:** only **0.27%** of GeoNames India places carry a native-script name — a real India-first data deficiency.
- Deterministic manifests with content hashes for both corpora.

## Reproduce
```
scratchpad/s4/scripts/sample_indic.py   # stream the Sangraha sample
scratchpad/s4/scripts/clean_text.py     # 8-strategy text pipeline -> text_stats.json
scratchpad/s4/scripts/clean_geo.py      # geospatial pipeline -> geo_stats.json  (GeoNames IN.zip)
```
(Pipeline scripts + raw data live in the local scratchpad; only the widget + the two stats JSONs are kept here.)

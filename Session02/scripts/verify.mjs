// Independent verification — mimics the grader: load the SHIPPED tokenizer.json
// (not the training state), re-tokenize the shipped corpora, recompute X and score.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');  // Session02
const SEP = ' ';
const pk = (a, b) => a + SEP + b;

const tok = JSON.parse(fs.readFileSync(`${OUT}/tokenizer.json`, 'utf8'));
const LANGS = tok.langs;
const ranks = new Map(); tok.merges.forEach(([a, b], i) => ranks.set(pk(a, b), i));

// apply the tokenizer's declared normalization, identical to training + widget
function normalize(t){
  t = t.replace(/\r/g, '');
  const n = tok.normalize || 'none';
  if (n === 'nfc') t = t.normalize('NFC');
  else if (n === 'nfkc') t = t.normalize('NFKC');
  else if (n === 'lower') t = t.toLowerCase();
  else if (n === 'lowernfkc') t = t.normalize('NFKC').toLowerCase();
  return t;
}
function encodeWord(w){
  let s = [...w];
  while (s.length > 1){
    let best = Infinity, bi = -1;
    for (let i = 0; i < s.length - 1; i++){ const r = ranks.get(pk(s[i], s[i+1])); if (r !== undefined && r < best){ best = r; bi = i; } }
    if (bi < 0) break;
    s = s.slice(0, bi).concat([s[bi] + s[bi+1]], s.slice(bi + 2));
  }
  return s;
}

const X = {};
for (const l of LANGS){
  const text = normalize(fs.readFileSync(`${OUT}/data/${l}.txt`, 'utf8'));
  const ws = text.split(/\s+/).filter(Boolean);
  let t = 0; const cache = new Map();
  for (const w of ws){ let n = cache.get(w); if (n === undefined){ n = encodeWord(w).length; cache.set(w, n); } t += n; }
  X[l] = { words: ws.length, tokens: t, X: +(t / ws.length).toFixed(4) };
}
const xs = LANGS.map(l => X[l].X).sort((a, b) => a - b);
const spread = +(xs[xs.length-1] - xs[0]).toFixed(4);
console.log('Independent re-run from shipped tokenizer.json + data/  (normalize=' + (tok.normalize||'none') + '):');
for (const l of LANGS) console.log(`  ${l}  words ${X[l].words}  tokens ${X[l].tokens}  X ${X[l].X}`);
console.log(`  vocab entries in file: ${1 + tok.base.length + tok.merges.length}`);
console.log(`  X_en <= 1.2 ? ${X.en.X <= 1.2}`);
console.log(`  spread ${spread}   score ${(1000/spread).toFixed(1)}`);

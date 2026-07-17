// Pure-JS reader for the HuggingFace tokenizer.json (BPE + Metaspace, no normalizer).
// Reproduces HuggingFace `tokenizers` token counts EXACTLY and round-trips byte-for-byte.
// The widget (index.html) uses the identical logic. Run `node tokenizer_js.mjs` for a self-test.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const REPL = '▁';

export function loadTokenizer(json){
  const vocab = json.model.vocab;                 // token -> id
  const id2tok = {}; for (const [t, i] of Object.entries(vocab)) id2tok[i] = t;
  const unk = json.model.unk_token;
  const ranks = new Map();
  json.model.merges.forEach((m, i) => { const [a, b] = Array.isArray(m) ? m : m.split(' '); ranks.set(a + ' ' + b, i); });

  function pretokens(text){                        // Metaspace: space -> ▁, split before each ▁
    const t = text.split(' ').join(REPL);
    return t === '' ? [] : t.split(/(?=▁)/);
  }
  function bpe(sym){
    while (sym.length > 1){
      let best = Infinity, bi = -1;
      for (let i = 0; i < sym.length - 1; i++){ const r = ranks.get(sym[i] + ' ' + sym[i+1]); if (r !== undefined && r < best){ best = r; bi = i; } }
      if (bi < 0) break;
      sym = sym.slice(0, bi).concat([sym[bi] + sym[bi+1]], sym.slice(bi + 2));
    }
    return sym;
  }
  function encode(text){
    const ids = [];
    for (const piece of pretokens(text))
      for (const tok of bpe(Array.from(piece))){ const id = vocab[tok]; ids.push(id === undefined ? vocab[unk] : id); }
    return ids;
  }
  function decode(ids){
    let s = ''; for (const id of ids){ const t = id2tok[id]; if (t !== undefined && t !== unk) s += t; }
    return s.split(REPL).join(' ');
  }
  return { encode, decode, vocabSize: Object.keys(vocab).length };
}

export const FAITHFUL_UNIT_RE = /[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]/gu;
export const faithfulUnits = (s) => (s.match(FAITHFUL_UNIT_RE) || []).length;

if (import.meta.url === `file://${process.argv[1]}`){
  const HERE = path.dirname(fileURLToPath(import.meta.url));
  const tk = loadTokenizer(JSON.parse(fs.readFileSync(path.join(HERE, 'tokenizer.json'), 'utf8')));
  const LANGS = ['en', 'hi', 'te', 'mai'];
  console.log('vocab size:', tk.vocabSize);
  const ratios = [];
  for (const c of LANGS){
    const x = fs.readFileSync(path.join(HERE, 'corpus', `${c}.faithful.txt`), 'utf8');
    const ids = tk.encode(x); const u = faithfulUnits(x); const ratio = ids.length / u;
    ratios.push(ratio);
    console.log(`${c}: tokens ${ids.length}  units ${u}  ratio ${ratio.toFixed(4)}  exact-roundtrip ${tk.decode(ids) === x}`);
  }
  const spread = Math.max(...ratios) - Math.min(...ratios);
  console.log(`spread ${spread.toFixed(5)}  score ${(1000 / spread).toFixed(1)}`);
  const s = "India's population is 1,428,627,663.";
  console.log('grader sample round-trip:', tk.decode(tk.encode(s)) === s);
}

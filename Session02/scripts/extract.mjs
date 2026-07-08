import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const HERE = path.dirname(fileURLToPath(import.meta.url));  // Session02/scripts
const dir = path.join(HERE, '..', 'data');                 // Session02/data
const langs = ['en','hi','te','mr'];
const meta = {};
for (const l of langs) {
  const j = JSON.parse(fs.readFileSync(`${dir}/${l}.json`, 'utf8'));
  const pages = j.query.pages;
  const page = pages[Object.keys(pages)[0]];
  let text = page.extract || '';
  const revid = page.revisions?.[0]?.revid;
  // Clean: drop Wikipedia section-header markup (runs of '='), keep the heading words;
  // normalize whitespace. Documented, reproducible preprocessing.
  text = text.replace(/\r/g, '');
  text = text.replace(/={2,}/g, ' ');      // "== Heading ==" -> " Heading "
  text = text.replace(/[ \t]+/g, ' ');
  text = text.replace(/\n{2,}/g, '\n').trim();
  fs.writeFileSync(`${dir}/${l}.txt`, text, 'utf8');
  const words = text.split(/\s+/).filter(Boolean);
  const uniq = new Set(words);
  const cps = [...text];
  const uniqCp = new Set(cps);
  meta[l] = { revid, title: page.title, chars: cps.length, words: words.length,
              uniqueWords: uniq.size, uniqueCodepoints: uniqCp.size,
              tail: text.slice(-120).replace(/\n/g,' ') };
}
console.log(JSON.stringify(meta, null, 2));

# DATASET_PIPELINE.md: Building the Dataset (Phase 1)

How to turn a list of companies into a train/valid/test dataset for LoRA
fine-tuning. Complete the environment setup in `SETUP.md` first (including
starting the local Qwen teacher server).

Guiding principle: **retrieval owns the facts, the model owns the form.**

Phase 1 has four steps:

```
companies.txt
   |  (1.1 retrieval)
   v
outputs/<company>-research-packet.json      evidence per company
   |  (1.2 labeling, local Qwen teacher)
   v
data/labeled/<company>.json                 draft gold labels (approved:false)
   |  (1.3 curation)
   v
data/labeled/<company>.json                 approved:true
   |  (1.4 assembly)
   v
data/mlx_brand_assistant/{train,valid,test}.jsonl   training dataset
```

---

## 1.1 Build evidence packets (retrieval)

Create `companies.txt` with one brand name per line (the repo ships a list of
150 DTC/consumer brands). Then run the hardened loop, which retries any
company that comes back with no evidence and paces requests to avoid
`ddgs` (DuckDuckGo) rate-limiting:

```bash
while IFS= read -r company; do
  [ -z "$company" ] && continue
  slug=$(echo "$company" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//;s/-$//')
  pkt="outputs/${slug}-research-packet.json"
  has_ev() { python3 -c "import json,sys; sys.exit(0 if len(json.load(open('$pkt')).get('evidence_passages',[]))>0 else 1)" 2>/dev/null; }
  if [ -f "$pkt" ] && has_ev; then echo "skip (has evidence): $company"; continue; fi
  for attempt in 1 2 3 4; do
    python run_brand_assistant.py --backend rag-only --company "$company" && has_ev && { echo "ok: $company"; break; }
    echo "empty/failed, retry $attempt for $company"; sleep 30
  done
  sleep 10
done < companies.txt
```

Check how many packets have real evidence:
```bash
python3 -c "import json,glob; c=[len(json.load(open(p)).get('evidence_passages',[]))>0 for p in glob.glob('outputs/*-research-packet.json')]; print(sum(c),'of',len(c),'packets have evidence')"
```

Drop any packet that still has no evidence (an empty packet cannot produce a
grounded label, and the teacher would hallucinate from it):
```bash
python3 -c "import json,glob,os; r=[os.remove(p) for p in glob.glob('outputs/*-research-packet.json') if len(json.load(open(p)).get('evidence_passages',[]))==0]; print('removed', len(r), 'empty packet(s)')"
```

Note: losing about 1-2% of companies to retrieval gaps is normal and has no
meaningful effect on the dataset. Grounded examples matter more than hitting
an exact count.

## 1.2 Generate labels (`make_labels.py`)

With the Qwen teacher server running (see `SETUP.md`), generate a draft label
for each packet:
```bash
python make_labels.py
```
For each packet, the local Qwen teacher reads the evidence and produces a
structured JSON answer (summary, audience, strategy, gaps, Instagram idea,
email, citations). Results go to `data/labeled/` with `"approved": false`.
The script skips already-labeled companies, so it resumes safely.

Safety check (remove any label that came from empty evidence):
```bash
grep -l "No evidence retrieved" data/labeled/*.json | xargs rm 2>/dev/null
```

## 1.3 Curate (`review_labels.py`)

Review each label and approve the good ones:
```bash
python review_labels.py
```
It shows one label at a time (summary, gaps, Instagram idea, email,
citations) and asks `approve? [y/n/o/q]`:
- `y` / Enter: approve (sets `"approved": true`)
- `n`: skip (fix the file later and re-run)
- `o`: print the file path to edit it
- `q`: quit (progress is saved)

What to check: gaps are evidence-backed and framed fairly
("strong on X, little visible Y"), citations point to real evidence, and the
email reads like something a person could send with minor edits. It is
resumable, so you can curate in batches.

## 1.4 Assemble the dataset (`build_dataset.py`)

Turn the approved labels into MLX-LM training files:
```bash
python build_dataset.py
```
Only `approved: true`, schema-valid labels are included. The split is
hash-based per company (stable across runs, no leakage between splits):
roughly 70% train, 10% validation, 20% held-out test.

Verify:
```bash
wc -l data/mlx_brand_assistant/*.jsonl
```

Phase 1 is complete once `data/mlx_brand_assistant/{train,valid,test}.jsonl`
exist with the expected counts.

---

## Scripts Reference

| Script | What it does |
|---|---|
| `run_brand_assistant.py` (`--backend rag-only`) | Searches the web for one company and writes its evidence packet. Used by the Phase 1.1 loop. |
| `make_labels.py` | Phase 1.2. Sends each packet's grounded prompt to the local Qwen teacher and writes a draft JSON label per company. |
| `review_labels.py` | Phase 1.3. Interactive curation: shows each label and lets you approve / skip / edit. |
| `build_dataset.py` | Phase 1.4. Assembles approved labels into train/valid/test JSONL with a stable held-out split. |
| `prepare_mlx_dataset.py` | Converts a legacy Colab JSONL dataset into MLX-LM format (optional seed data). |
| `companies.txt` | Input list of brand names, one per line. You supply this; the system fetches the research. |

---

## Generated Files Reference

### `outputs/<company>-research-packet.json` (one per company, from 1.1)
The retrieved evidence for a company. Fields:
- `company`, `person`: the inputs.
- `profiles`: known official links (website, Instagram, etc.).
- `queries`: the search queries that were run.
- `sources`: the web sources found, with titles and URLs.
- `evidence_passages`: the extracted, ranked text passages (the actual evidence).
- `grounded_prompt`: the compiled prompt, including the numbered Evidence block that citations refer to.
- `cache_dir`: where fetched pages are cached.

### `data/labeled/<company>.json` (one per company, from 1.2/1.3)
A draft or approved training label. Fields:
- `company`, `person`: the inputs.
- `grounded_prompt`: the input evidence prompt (the training "prompt").
- `completion`: the teacher's JSON answer as a string (the training target). Inside it:
  - `brand_summary`, `likely_target_audience`
  - `current_marketing_strategy` (list)
  - `marketing_gaps` (list)
  - `instagram_post_idea` (`hook`, `concept`, `why_it_closes_the_gap`)
  - `email_subject`, `email_body`
  - `citations` (e.g. `["[1]"]`), referencing items in the grounded prompt's Evidence block
- `validation_errors`: `[]` means schema-valid (structure only, not grounding).
- `approved`: `false` until you approve it in curation.

### `data/mlx_brand_assistant/{train,valid,test}.jsonl` (from 1.4)
The final training dataset. One JSON object per line:
- `prompt`: the grounded prompt (evidence in, numbered for citations).
- `completion`: the approved JSON answer (the target the model learns to produce).

This is what Phase 2 (`mlx_lm.lora`) trains on.

---

## Phase 1 Issues and Notes

**"No evidence retrieved" / 0 evidence passages.** `ddgs` rate-limiting,
usually from many fast searches. Transient. The hardened loop above paces and
retries; the delete step removes any that stay empty. Never label an
empty-evidence packet: with no evidence the teacher produces a confident but
ungrounded answer (a hallucination).

**If you use a CLOUD teacher instead of local Qwen** (optional path):
- `HTTP 401 Unauthorized`: the API key is wrong or a placeholder. Check length
  with `echo ${#TEACHER_API_KEY}` (a real key is ~51+ chars).
- `HTTP 429 Too Many Requests` on the first call: usually no API credits, not
  rate-limiting. A ChatGPT Plus subscription does NOT include API credits.

---

## Known Limitations

- **Citations are coarse-grained.** The teacher tends to cite a single
  evidence item, sometimes repeated to fill the citation field
  (e.g. `["[2]", "[2]"]`). The citations are grounded (they point to real,
  supporting evidence) but not fine-grained per claim. This is acceptable and
  can be deduplicated; richer multi-source citing would require a stronger
  teacher or a tightened labeling prompt.
- **Schema validation is structural only.** `validation_errors: []` means the
  JSON is well-formed, not that the claims are grounded. Human curation
  (Phase 1.3) is the real quality gate.
- **Empty-evidence packets must be dropped before labeling** (see above).

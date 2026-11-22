#!/bin/bash
# Phase 1.1 - retrieve evidence for every company in companies.txt.
# A company counts as done only when its packet contains evidence; empty results are retried.
# Usage (repo root, environment active):  bash retrieve_all.sh
has_evidence() {
  python3 -c "import json,sys; sys.exit(0 if len(json.load(open(sys.argv[1])).get('evidence_passages',[]))>0 else 1)" "$1" 2>/dev/null
}
while IFS= read -r company; do
  [ -z "$company" ] && continue
  slug=$(echo "$company" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//;s/-$//')
  packet="outputs/${slug}-research-packet.json"
  if [ -f "$packet" ] && has_evidence "$packet"; then echo "skip (has evidence): $company"; continue; fi
  for attempt in 1 2 3 4; do
    python run_brand_assistant.py --backend rag-only --company "$company" >/dev/null 2>&1
    if [ -f "$packet" ] && has_evidence "$packet"; then echo "ok: $company"; break; fi
    echo "no evidence yet (attempt $attempt): $company"; sleep 30
  done
  sleep 10
done < companies.txt
python3 -c "
import json,glob
p=glob.glob('outputs/*-research-packet.json'); n=sum(len(json.load(open(f)).get('evidence_passages',[]))>0 for f in p)
print(f'{n} of {len(p)} packets have evidence; the rest will get no label')"

import json
from pathlib import Path

source = Path("data/v3_datasets/v3-pilot-2026-11-22/train.jsonl")
output = Path("data/v3_smoke")

if not source.exists():
    raise FileNotFoundError(f"Missing dataset: {source}")

records = [
    json.loads(line)
    for line in source.read_text().splitlines()
    if line.strip()
]

assert len(records) == 7, f"Expected 7 samples, got {len(records)}"

output.mkdir(parents=True, exist_ok=True)

# Fixed split for reproducible smoke testing.
train = [record for i, record in enumerate(records) if i != 3]
valid = [records[3]]

for name, samples in [("train", train), ("valid", valid)]:
    with (output / f"{name}.jsonl").open("w") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

# MLX-LM should not read an empty test.jsonl.
(output / "test.jsonl").unlink(missing_ok=True)

print(f"Prepared {len(train)} train and {len(valid)} validation samples.")

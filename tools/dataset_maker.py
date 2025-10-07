import json
import random
from pathlib import Path

# ====== CONFIG ======
DATASET_ROOT = Path("datasets/medquad")   # adjust if needed
SEED = 42
FRACTION = 0.45   # fraction of the combined unique examples to MOVE into part3
# If you prefer a fixed size, set MAX_COUNT (int) and FRACTION=None
MAX_COUNT = None

# (src_part1,                src_part2,                                  out_client_dir,              out_filename)
JOBS = [
    (("client_0", "gard_qa_part1.json"),
     ("client_1", "gard_qa_part2.json"),
     ("client_8", "gard_qa_part3.json")),

    (("client_2", "ghr_qa_part1.json"),
     ("client_3", "ghr_qa_part2.json"),
     ("client_9", "ghr_qa_part3.json")),

    (("client_4", "general_health_qa_part1.json"),
     ("client_5", "general_health_qa_part2.json"),
     ("client_10", "general_health_qa_part3.json")),

    (("client_6", "internal_medicine_qa_part1.json"),
     ("client_7", "internal_medicine_qa_part2.json"),
     ("client_11", "internal_medicine_qa_part3.json")),
]
# ====================

random.seed(SEED)

def _load_list(p: Path):
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{p} must contain a JSON list")
    return data

def _dump_list(p: Path, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _norm_pair(rec):
    # normalize whitespace to make dedup safer
    q = (rec.get("question") or "").strip()
    a = (rec.get("answer") or "").strip()
    return q, a

for (c1, f1), (c2, f2), (out_c, out_f) in JOBS:
    src1 = DATASET_ROOT / c1 / f1
    src2 = DATASET_ROOT / c2 / f2
    outp = DATASET_ROOT / out_c / out_f

    part1 = _load_list(src1)
    part2 = _load_list(src2)

    # Tag origin so we can remove moved items from the correct file
    tagged = []
    seen = set()

    for rec in part1:
        key = _norm_pair(rec)
        if key not in seen:
            seen.add(key)
            tagged.append(("p1", rec, key))

    for rec in part2:
        key = _norm_pair(rec)
        if key not in seen:
            seen.add(key)
            tagged.append(("p2", rec, key))

    # Decide how many to move
    random.shuffle(tagged)
    if MAX_COUNT is not None:
        k = min(MAX_COUNT, len(tagged))
    else:
        k = max(1, int(len(tagged) * FRACTION))

    moved = tagged[:k]         # to part3
    kept  = tagged[k:]         # stay in p1/p2

    # Rebuild part1/part2 without moved items
    moved_keys = {t[2] for t in moved}
    new_part1, new_part2 = [], []
    for origin, rec, key in kept:
        if origin == "p1":
            new_part1.append(rec)
        else:
            new_part2.append(rec)

    # Sanity: no overlap across outputs
    def _keys(lst): return {_norm_pair(r) for r in lst}
    assert not _keys(new_part1) & _keys(new_part2), "Overlap between remaining part1 and part2!"
    part3 = [rec for _, rec, _ in moved]
    assert not _keys(new_part1) & _keys(part3), "Overlap between part1 and part3!"
    assert not _keys(new_part2) & _keys(part3), "Overlap between part2 and part3!"

    # Write back
    _dump_list(src1, new_part1)
    _dump_list(src2, new_part2)
    _dump_list(outp, part3)

    print(f"[OK] {src1.name}: {len(part1)} -> {len(new_part1)}")
    print(f"[OK] {src2.name}: {len(part2)} -> {len(new_part2)}")
    print(f"[OK] {outp} created with {len(part3)} moved records\n")

print("Done. Update your config 'num_of_clients' accordingly (e.g., 11).")

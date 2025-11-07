# Usage:
#   python make_medmentions.py --out-dir datasets/medmentions_st21pv_ner --num-clients 10 --chunk-size 0
#
# Notes:
# - We load the KB schema (medmentions_st21pv_bigbio_kb) and convert to a NER-like format.
# - Each passage (title/abstract) becomes one example with tokens + BIO tags.
# - Tokens are whitespace-based; BIO tags come from entity character offsets.
# - We shard by document_id using a balanced round-robin to get even client sizes.

import argparse, json, os, re
from collections import defaultdict
from typing import List, Tuple, Dict
from datasets import load_dataset
import re

try:
    import nltk
    _HAS_NLTK = True
    try:
        # Check if 'punkt' tokenizer is already installed
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        # If not installed, download it
        nltk.download("punkt")
except Exception:
    _HAS_NLTK = False

SENT_SEP = re.compile(r"(?<=[.!?])\s+")

def sentence_spans(text):
    """Return list of (start, end) char spans for sentences."""
    if _HAS_NLTK:
        sent_tokens = nltk.tokenize.sent_tokenize(text)
        spans, cursor = [], 0
        for s in sent_tokens:
            i = text.find(s, cursor)
            spans.append((i, i + len(s)))
            cursor = i + len(s)
        return spans
    # fallback regex splitter
    parts = SENT_SEP.split(text)
    spans, cursor = [], 0
    for s in parts:
        i = text.find(s, cursor)
        spans.append((i, i + len(s)))
        cursor = i + len(s)
    return spans

def kb_record_to_ner_examples(rec, granularity="sentence"):
    """
    Convert one KB record (document) into NER examples.
    granularity: "sentence" (recommended) or "passage".
    """
    doc_id = str(rec["document_id"])
    # collect (etype, [(start,end), ...]) in global coords
    entities = []
    for e in rec.get("entities", []):
        etype = e["type"][0] if isinstance(e["type"], list) and e["type"] else e["type"]
        spans = [(int(s), int(t)) for (s, t) in e.get("offsets", [])]
        entities.append((str(etype), spans))

    out = []
    for p in rec.get("passages", []):
        p_text = " ".join(p.get("text", []) or [])
        if not p_text:
            continue
        p_offs = p.get("offsets", [])
        p0 = int(p_offs[0][0]) if p_offs else 0

        # choose segments: one per passage or one per sentence
        if granularity == "passage":
            segs = [(0, len(p_text))]
        else:
            # sentence-relative spans in passage coords
            segs = sentence_spans(p_text)

        for (s0, s1) in segs:
            seg_text = p_text[s0:s1]
            # tokenize segment with char spans
            tokens, tok_spans = [], []
            for m in re.finditer(r"\S+", seg_text):
                tokens.append(m.group(0))
                # map to passage-local, then to segment-local
                tok_spans.append((m.start(), m.end()))
            tags = ["O"] * len(tokens)

            # tag tokens by intersecting global entity spans with this segment
            seg_global_start = p0 + s0
            seg_global_end   = p0 + s1

            # collect entity spans in segment-local coords
            for etype, espans in entities:
                loc = []
                for (es, ee) in espans:
                    if es < seg_global_end and ee > seg_global_start:
                        ls = max(0, es - seg_global_start)
                        le = min(s1 - s0, ee - seg_global_start)
                        if ls < le:
                            loc.append((ls, le))
                if not loc:
                    continue
                # label BIO for contiguous token runs that intersect any loc span
                covered = []
                for i, (ts, te) in enumerate(tok_spans):
                    if any(ts < le and ls < te for (ls, le) in loc):
                        covered.append(i)
                if not covered:
                    continue
                # collapse into runs
                run = [covered[0]]
                for i in covered[1:]:
                    if i == run[-1] + 1:
                        run.append(i)
                    else:
                        tags[run[0]] = f"B-{etype}"
                        for j in run[1:]:
                            if tags[j] == "O":
                                tags[j] = f"I-{etype}"
                        run = [i]
                tags[run[0]] = f"B-{etype}"
                for j in run[1:]:
                    if tags[j] == "O":
                        tags[j] = f"I-{etype}"

            out.append({
                "id": f"{rec['id']}:{p['id']}:{s0}-{s1}",
                "document_id": doc_id,
                "text": seg_text,
                "tokens": tokens,
                "ner_tags": tags,
            })
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="bigbio/medmentions",
                    help="HF dataset repo id")
    ap.add_argument("--config", default="medmentions_st21pv_bigbio_kb",
                    help="KB config; e.g., medmentions_st21pv_bigbio_kb")
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="If >0, write split into *_partN.json chunks of this many records.")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

TOKEN_PATTERN = re.compile(r"\S+")

def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int,int]]]:
    """Whitespace-ish tokenization; returns tokens and (start,end) char spans in 'text'."""
    tokens, spans = [], []
    for m in TOKEN_PATTERN.finditer(text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans

def intersect(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    """Closed-open interval intersection check for character spans."""
    return a[0] < b[1] and b[0] < a[1]

def label_tokens_bio(token_spans: List[Tuple[int,int]],
                     entity_spans_local: List[Tuple[int,int]],
                     label: str,
                     tags: List[str]):
    """
    Apply BIO tags to tokens whose spans intersect any of the entity spans.
    entity_spans_local: list of (start,end) in the *passage-local* coordinates
    """
    covered_idxs = []
    for i, ts in enumerate(token_spans):
        if any(intersect(ts, es) for es in entity_spans_local):
            covered_idxs.append(i)
    if not covered_idxs:
        return
    # Collapse to contiguous runs and label each run with B- / I-
    run = [covered_idxs[0]]
    for i in covered_idxs[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            # Flush previous run
            if tags[run[0]] == "O":
                tags[run[0]] = f"B-{label}"
                for j in run[1:]:
                    if tags[j] == "O":
                        tags[j] = f"I-{label}"
            run = [i]
    # Flush last run
    if tags[run[0]] == "O":
        tags[run[0]] = f"B-{label}"
        for j in run[1:]:
            if tags[j] == "O":
                tags[j] = f"I-{label}"

def passage_examples_from_kb_record(rec) -> List[Dict]:
    """
    Convert a KB record (one document) into NER-like examples per passage.
    We keep offsets local to each passage using passage['offsets'][0] as the base.
    """
    doc_id = str(rec["document_id"])
    examples = []
    # Build quick access of entities' spans and types (global coordinates)
    entities = []
    for e in rec.get("entities", []):
        etype = e["type"][0] if isinstance(e["type"], list) and e["type"] else e["type"]
        # Each entity may have one or more (start,end) spans (global)
        spans = [(int(s), int(t)) for (s, t) in e.get("offsets", [])]
        entities.append((etype, spans))

    # Iterate passages (title/abstract)
    for p in rec.get("passages", []):
        # 'text' is usually a list of strings; join them with a single space
        p_text = " ".join(p.get("text", []) or [])
        if not p_text:
            continue
        # passage offsets are typically a list with one (start,end) pair
        p_offsets = p.get("offsets", [])
        if not p_offsets:
            # No offsets -> treat as local [0, len)
            p_start = 0
        else:
            p_start = int(p_offsets[0][0])

        tokens, tok_spans = tokenize_with_spans(p_text)
        tags = ["O"] * len(tokens)

        # Map any entity span that overlaps this passage into passage-local coordinates
        p_end = p_start + len(p_text)
        for etype, espans in entities:
            # collect local spans for this passage
            local_spans = []
            for (es, ee) in espans:
                if es < p_end and ee > p_start:  # intersects passage
                    ls = max(0, es - p_start)
                    le = min(len(p_text), ee - p_start)
                    if ls < le:
                        local_spans.append((ls, le))
            if local_spans:
                label_tokens_bio(tok_spans, local_spans, str(etype), tags)

        examples.append({
            "id": f"{rec['id']}:{p['id']}",
            "document_id": doc_id,
            "text": p_text,
            "tokens": tokens,
            "ner_tags": tags,
        })
    return examples

def balanced_round_robin_map(all_doc_ids: List[str], num_clients: int) -> Dict[str, int]:
    ordered = sorted(set(all_doc_ids))
    return {doc_id: i % num_clients for i, doc_id in enumerate(ordered)}

def write_json(records, base_path, chunk_size: int):
    if chunk_size and chunk_size > 0:
        part = 1
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i+chunk_size]
            with open(f"{base_path}_part{part}.json", "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            part += 1
    else:
        with open(f"{base_path}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    print(f"Loading {args.dataset_name} :: {args.config}")
    ds = load_dataset(args.dataset_name, name=args.config)  # KB schema

    # Pick validation key
    split_keys = list(ds.keys())
    val_key = "validation" if "validation" in split_keys else ("dev" if "dev" in split_keys else None)
    if val_key is None:
        raise RuntimeError(f"Expected 'validation' or 'dev' in splits, got: {split_keys}")

    splits = {"train": ds["train"], "validation": ds[val_key], "test": ds["test"]}

    # Convert KB -> NER-like examples (per passage)
    ner_splits = {k: [] for k in splits}
    all_doc_ids = []
    GRANULARITY = "sentence"  # or "passage"

    for split_name, split_ds in splits.items():
        for rec in split_ds:
            all_doc_ids.append(str(rec["document_id"]))
            ner_splits[split_name].extend(kb_record_to_ner_examples(rec, granularity=GRANULARITY))

    # Build balanced doc_id -> client map across ALL docs
    doc_to_client = balanced_round_robin_map(all_doc_ids, args.num_clients)

    # Bucket per client per split
    clients = {
        cid: {"train": [], "validation": [], "test": []}
        for cid in range(args.num_clients)
    }
    for split_name, examples in ner_splits.items():
        for ex in examples:
            cid = doc_to_client[str(ex["document_id"])]
            clients[cid][split_name].append(ex)

    # Write to disk
    for cid in range(args.num_clients):
        cdir = os.path.join(args.out_dir, f"client_{cid}")
        ensure_dir(cdir)
        for split_name in ("train", "validation", "test"):
            base = os.path.join(cdir, split_name)
            write_json(clients[cid][split_name], base, args.chunk_size)
        sizes = {k: len(v) for k, v in clients[cid].items()}
        print(f"[client_{cid}] train={sizes['train']} val={sizes['validation']} test={sizes['test']}")

    total_train = sum(len(clients[c]["train"]) for c in clients)
    total_val   = sum(len(clients[c]["validation"]) for c in clients)
    total_test  = sum(len(clients[c]["test"]) for c in clients)
    print(f"TOTAL -> train={total_train} validation={total_val} test={total_test}")
    print("Done.")

if __name__ == "__main__":
    main()

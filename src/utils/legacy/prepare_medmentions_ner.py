import argparse, json, os, hashlib, random
from collections import Counter
from datasets import load_dataset
from transformers import GPT2TokenizerFast

SEED = 17
random.seed(SEED)

def md5int(s): return int(hashlib.md5(s.encode()).hexdigest(), 16)
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def get_text(ex):
    # BigBio KB schema: passages is a list of dicts; each p["text"] is a LIST[str]
    # We first join within a passage, then join passages with a single space.
    parts = []
    for p in ex["passages"]:
        t = p.get("text", "")
        if isinstance(t, list):
            parts.append(" ".join(t))
        elif isinstance(t, str):
            parts.append(t)
        else:
            # fallback if it's None or unexpected
            parts.append("")
    return " ".join(parts)


def build_label_set(split):
    # FULL uses UMLS semantic types (many more than 21). We use semantic_type_id as labels.
    stypes = set()
    for ex in split:
        for ent in ex["entities"]:
            for t in ent.get("type", []):  # medmentions.py maps semantic_type_id -> "type"
                stypes.add(t)
    return sorted(stypes)

def dom_sem_type(ex):
    c = Counter()
    for ent in ex["entities"]:
        for (s,e) in ent["offsets"]:
            if ent.get("type"):
                c[ent["type"][0]] += (e-s)
    return c.most_common(1)[0][0] if c else "NONE"

def assign_client(doc_id, ex, strategy, nclients):
    return md5int(doc_id) % nclients if strategy=="iid" else md5int(dom_sem_type(ex)) % nclients

def to_bio_gpt2(text, entities, tok, max_len):
    enc = tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len
    )
    offs = enc["offset_mapping"]

    chartype = [None]*(len(text)+1)
    for ent in entities:
        t = (ent.get("type") or ["ENT"])[0]
        for s,e in ent["offsets"]:
            for i in range(s, min(e, len(chartype))):
                if chartype[i] is None:
                    chartype[i] = t

    tags, prev = [], None
    from collections import Counter
    for (s,e) in offs:
        if s == e:
            tags.append("O"); prev = None; continue
        span = [chartype[i] for i in range(s, min(e, len(chartype))) if chartype[i] is not None]
        t = Counter(span).most_common(1)[0][0] if span else None
        if t is None: tags.append("O"); prev = None
        else: tags.append(("B-" if t != prev else "I-")+t); prev = t
    return tags, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/medmentions")
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--shard_strategy", choices=["iid","stype"], default="iid",
                    help="iid: hash doc id; stype: hash dominant semantic type (label-skew)")
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    ensure_dir(args.root)
    print("Loading BigBio MedMentions FULL...")
    ds = load_dataset("bigbio/medmentions", name="medmentions_full_bigbio_kb")  # FULL variant

    # label space = ALL semantic types in FULL split
    label_set = build_label_set(ds["train"])
    bio_labels = ["O"] + [p+t for t in label_set for p in ("B-","I-")]
    label2id = {l:i for i,l in enumerate(bio_labels)}
    with open(os.path.join(args.root,"label_list.json"),"w") as f: json.dump(bio_labels,f,indent=2)
    with open(os.path.join(args.root,"label2id.json"),"w") as f: json.dump(label2id,f,indent=2)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    max_len = min(args.max_length, 1024)
    def writers(split):
        ws=[]
        for k in range(args.num_clients):
            cdir = os.path.join(args.root, f"client_{k}"); ensure_dir(cdir)
            ws.append(open(os.path.join(cdir, f"{split}.jsonl"), "w", encoding="utf-8"))
        return ws

    W = {sp: writers(sp) for sp in ["train","val","test"]}

    for sp_name, split in [("train", ds["train"]), ("val", ds["validation"]), ("test", ds["test"])]:
        for ex in split:
            doc_id = ex["document_id"]
            text = get_text(ex)
            tags, enc = to_bio_gpt2(text, ex["entities"], tok, max_len)
            cid = assign_client(doc_id, ex, args.shard_strategy, args.num_clients)
            rec = {
                "document_id": doc_id,
                "text": text,
                "labels_str": tags,
                "tokenizer": "gpt2",
                "tokenizer_kwargs": {"max_length": max_len, "truncation": True, "add_special_tokens": True},
                "max_length_hint": max_len
            }
            W[sp_name][cid].write(json.dumps(rec)+"\n")

    for sp in W.values():
        for fh in sp: fh.close()
    print("Done →", args.root)

if __name__ == "__main__":
    main()

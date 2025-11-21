import torch

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int  # [start, end)
    label: str  # semantic type (e.g., T017)
    entity_id: Optional[str] = None


def _bio_to_spans(tags: List[str], doc_id: str) -> List[Span]:
    spans, i, L = [], 0, len(tags)
    while i < L:
        tag = tags[i]
        if not tag or tag == "O":
            i += 1;
            continue
        prefix, label = (tag.split("-", 1) + [""])[:2] if "-" in tag else ("B", tag)
        j = i + 1
        while j < L and tags[j].startswith("I-") and tags[j].split("-", 1)[1] == label:
            j += 1
        spans.append(Span(doc_id, i, j, label))
        i = j
    return spans


class StrictMentionAndDocEvaluator:
    """Mention-level: exact span+label. Document-level: set of labels per doc."""

    def __init__(self, id2label: Dict[int, str], label_only: bool = True):
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label_only = label_only
        self.tp_m = self.fp_m = self.fn_m = 0
        self.tp_d = self.fp_d = self.fn_d = 0
        self.gold_doc_labels: Dict[str, set] = {}
        self.pred_doc_labels: Dict[str, set] = {}

    @staticmethod
    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    def _mkey(self, s: Span):  # what defines a mention match
        return (s.doc_id, s.start, s.end, s.label)

    def _doclab(self, s: Span):
        return s.label

    def update_batch(self, logits: torch.Tensor, labels: torch.Tensor,
                     doc_ids: List[str], word_lengths: List[int]):
        pred_ids = logits.argmax(-1)  # [B,T]
        B, T = labels.shape
        for b in range(B):
            doc_id = str(doc_ids[b])
            gold_tags, pred_tags, seen = [], [], 0
            for t in range(T):
                if labels[b, t].item() != -100:  # take only first subword positions
                    gold_tags.append(self.id2label[int(labels[b, t].item())])
                    pred_tags.append(self.id2label[int(pred_ids[b, t].item())])
                    seen += 1
                    if seen == word_lengths[b]:
                        break
            g_spans = _bio_to_spans(gold_tags, doc_id)
            p_spans = _bio_to_spans(pred_tags, doc_id)
            g_keys = {self._mkey(s) for s in g_spans}
            p_keys = {self._mkey(s) for s in p_spans}
            self.tp_m += len(g_keys & p_keys)
            self.fp_m += len(p_keys - g_keys)
            self.fn_m += len(g_keys - p_keys)
            g_set = {self._doclab(s) for s in g_spans}
            p_set = {self._doclab(s) for s in p_spans}
            self.gold_doc_labels.setdefault(doc_id, set()).update(g_set)
            self.pred_doc_labels.setdefault(doc_id, set()).update(p_set)

    def finalize(self) -> Dict[str, float]:
        pm, rm, f1m = self._prf(self.tp_m, self.fp_m, self.fn_m)
        for d in set(self.gold_doc_labels) | set(self.pred_doc_labels):
            g, p = self.gold_doc_labels.get(d, set()), self.pred_doc_labels.get(d, set())
            self.tp_d += len(g & p);
            self.fp_d += len(p - g);
            self.fn_d += len(g - p)
        pd, rd, f1d = self._prf(self.tp_d, self.fp_d, self.fn_d)
        return {
            "mention_precision": pm, "mention_recall": rm, "mention_f1": f1m,
            "document_precision": pd, "document_recall": rd, "document_f1": f1d,
            "tp_m": self.tp_m, "fp_m": self.fp_m, "fn_m": self.fn_m,
            "tp_d": self.tp_d, "fp_d": self.fp_d, "fn_d": self.fn_d,
        }

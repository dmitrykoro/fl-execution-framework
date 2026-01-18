"""
NER Metrics for MedMentions-style BIO Tagging
=============================================

This module provides utilities to compute **strict mention-level** and
**document-level** metrics for Named Entity Recognition (NER) in setups similar
to MedMentions (ST21pv). It assumes BIO tagging and that model outputs/labels
may include subword positions, where only the first subword location is valid.

Two evaluation granularities are implemented:

1) Strict Mention-level (exact span + label match)
   - A predicted entity counts as True Positive (TP) only if its start index,
     end index (exclusive), and label exactly match a gold entity.
   - Otherwise, it contributes to FP (if predicted but not gold) or FN (if gold
     but not predicted).

2) Document-level label set
   - For each document, we collect the set of unique labels present in gold and
     predicted mentions. Matching is based on **labels only** (not spans).
   - Precision/Recall/F1 computed over set membership across all documents.

This module is primarily designed for NER experiments on MedMentions-like
datasets, but it is generic enough to be reused for any BIO-tagged sequence
labeling task with the same conventions.

Notes
-----
- Label IDs are mapped to string tags via ``id2label``.
- The label ID ``-100`` is treated as "ignore" (typical for first-subword-only
  supervision with tokenizers); only positions != -100 are considered.
- ``word_lengths[b]`` specifies how many *word-level* (first-subword) positions
  are present in the b-th sequence; iteration stops after that many valid
  positions have been processed.
"""
import torch

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass(frozen=True)
class Span:
    """A contiguous entity span in word-level indices.

        Attributes:
            doc_id: Identifier for the document the span belongs to.
            start: Start index (inclusive) in word-level coordinates.
            end: End index (exclusive) in word-level coordinates.
            label: Entity label (e.g., semantic type like ``T017``).
            entity_id: Optional unique identifier for linking (unused here).
    """
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
        """Accumulate counts for a batch of model outputs and gold labels.

        Args:
            logits: Float tensor of shape ``[B, T, C]`` with class scores.
            labels: Long tensor of shape ``[B, T]`` with label IDs.
                Positions with value ``-100`` are ignored (non-first subwords).
            doc_ids: List of document IDs, length ``B``.
            word_lengths: List of valid word-level lengths per sequence (how
                many non-ignored positions to consider in each row).

        Behavior:
            - Converts IDs to tag strings using ``id2label``.
            - Truncates each sequence after ``word_lengths[b]`` *valid* positions
              (skipping ignored positions).
            - Builds gold and predicted spans via BIO parsing.
            - Updates mention-level TP/FP/FN and document-level per-doc label sets.

        Returns:
            None. Internal counters are updated in-place.
        """
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
        """Compute and return precision/recall/F1 for mention- and doc-level.

        After all batches have been processed with :meth:`update_batch`, call
        :meth:`finalize` once to aggregate document-level counts and produce the
        final metrics. This method is idempotent with respect to additional
        calls (but repeated calls will re-accumulate document-level TP/FP/FN
        from the stored sets).

        Returns:
            Dict[str, float]: A dictionary containing:
                - "mention_precision", "mention_recall", "mention_f1"
                - "document_precision", "document_recall", "document_f1"
                - "tp_m", "fp_m", "fn_m"  (mention-level counts)
                - "tp_d", "fp_d", "fn_d"  (document-level counts)
        """
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

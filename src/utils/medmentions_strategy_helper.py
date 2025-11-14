from typing import Dict, List, Tuple, Set

from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy


def aggregate_strict_medmentions_metrics(
    *,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    removed_client_ids: Set[str],
    remove_clients: bool,
    current_round: int,
    begin_removing_from_round: int,
    strategy_history,
) -> Dict[str, Scalar]:
    """Compute optional strict MedMentions micro-averaged metrics.

    - Uses only non-removed clients once removal has started.
    - Writes mention/document precision/recall/F1 into `strategy_history.rounds_history`
      if those lists exist.
    - Returns a metrics dict suitable for the strategy's `metrics_aggregated`.
    """

    # Only enable if at least one client reported strict counts (NER/GPT-2 case)
    has_strict = any(
        ("tp_m" in ev.metrics) or ("tp_d" in ev.metrics) for _, ev in results
    )

    metrics_aggregated: Dict[str, Scalar] = {}
    if not has_strict:
        return metrics_aggregated

    if remove_clients and current_round >= begin_removing_from_round:
        effective_results = [
            (cp, ev) for (cp, ev) in results if cp.cid not in removed_client_ids
        ]
    else:
        effective_results = results

    # Aggregate strict mention-level and document-level counts
    sum_tp_m = sum(int(ev.metrics.get("tp_m", 0)) for _, ev in effective_results)
    sum_fp_m = sum(int(ev.metrics.get("fp_m", 0)) for _, ev in effective_results)
    sum_fn_m = sum(int(ev.metrics.get("fn_m", 0)) for _, ev in effective_results)

    sum_tp_d = sum(int(ev.metrics.get("tp_d", 0)) for _, ev in effective_results)
    sum_fp_d = sum(int(ev.metrics.get("fp_d", 0)) for _, ev in effective_results)
    sum_fn_d = sum(int(ev.metrics.get("fn_d", 0)) for _, ev in effective_results)

    def _prf(tp: int, fp: int, fn: int):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        return p, r, f1

    mp, mr, mf1 = _prf(sum_tp_m, sum_fp_m, sum_fn_m)
    dp, dr, df1 = _prf(sum_tp_d, sum_fp_d, sum_fn_d)

    # Persist to rounds history only if those fields exist
    rh = getattr(strategy_history, "rounds_history", None)
    if rh is not None and hasattr(rh, "mention_precision_history"):
        rh.mention_precision_history.append(mp)
        rh.mention_recall_history.append(mr)
        rh.mention_f1_history.append(mf1)
        rh.document_precision_history.append(dp)
        rh.document_recall_history.append(dr)
        rh.document_f1_history.append(df1)

    # Expose in aggregated metrics output
    metrics_aggregated.update({
        "mention_precision": float(mp),
        "mention_recall": float(mr),
        "mention_f1": float(mf1),
        "document_precision": float(dp),
        "document_recall": float(dr),
        "document_f1": float(df1),
    })

    return metrics_aggregated

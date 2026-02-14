"""
erag_mod.py
Evaluates retrieval quality using IR metrics (P@k, recall, nDCG, MRR, etc.)
with relevance determined by downstream performance (EM, F1, etc.) on each document.

1. For each query-doc: generate an answer using only that document
2. Compute downstream score (EM/F1) → relevance label [0,1]
3. Build qrel (gold relevance) and run (ranking)
4. Compute IR metrics via pytrec_eval

MOD:
- "ID-mode" support to return real (query_id, doc_id, score) triplets.
  In ID-mode:
    retrieval_results: Dict[query_id, List[doc_id]]
    expected_outputs: Dict[query_id, List[str]]
    query_id_to_query: Dict[query_id, query_text]
    doc_id_to_document: Dict[doc_id, doc_text]
    inputs_are_ids=True
- Also returns `triples`: list of dicts {"query_id","doc_id","score"}.
"""

from typing import Dict, Callable, List, Union, Set, Optional, Any
from collections import Counter

import pytrec_eval


def eval(
    retrieval_results: Dict[str, List[str]],
    expected_outputs: Dict[str, List[str]],
    text_generator: Callable[[Dict[str, List[str]]], Dict[str, str]],
    downstream_metric: Callable[[Dict[str, str], Dict[str, List[str]]], Dict[str, Union[int, float]]],
    retrieval_metrics: Set[str],
    *,
    inputs_are_ids: bool = False,
    query_id_to_query: Optional[Dict[str, str]] = None,
    doc_id_to_document: Optional[Dict[str, str]] = None,
    **args: Any,
):
    """
    Args (legacy mode, inputs_are_ids=False):
        retrieval_results: {query_text: [doc_text1, doc_text2, ...]}
        expected_outputs: {query_text: [gold1, gold2, ...]}

    Args (ID-mode, inputs_are_ids=True):
        retrieval_results: {query_id: [doc_id1, doc_id2, ...]}
        expected_outputs: {query_id: [gold1, gold2, ...]}
        query_id_to_query: {query_id: query_text}
        doc_id_to_document: {doc_id: doc_text}
    """

    assert set(retrieval_results.keys()) == set(expected_outputs.keys()), (
        "The keys in retrieval results and expected outputs do not match."
    )

    # -----------------------------
    # Compute downstream scores per (query, doc)
    # -----------------------------
    if inputs_are_ids:
        if query_id_to_query is None or doc_id_to_document is None:
            raise ValueError(
                "In ID-mode you must provide both query_id_to_query and doc_id_to_document."
            )

        # evaluation_scores[qid][doc_id] = score
        evaluation_scores: Dict[str, Dict[str, Union[int, float]]] = {qid: {} for qid in retrieval_results.keys()}

        max_len = max((len(lst) for lst in retrieval_results.values()), default=0)

        for i in range(max_len):
            # build items for rank i
            # each item: (qid, query_text, doc_id, doc_text)
            items = []
            for qid, doc_ids in retrieval_results.items():
                if i >= len(doc_ids):
                    continue

                if qid not in query_id_to_query:
                    raise KeyError(f"query_id_to_query missing query_id: {qid}")

                doc_id = str(doc_ids[i])
                if doc_id not in doc_id_to_document:
                    raise KeyError(f"doc_id_to_document missing doc_id: {doc_id}")

                qtext = query_id_to_query[qid]
                dtext = doc_id_to_document[doc_id]
                items.append((qid, qtext, doc_id, dtext))

            if not items:
                continue

            # We can batch-generate only for distinct query_text keys in the same call
            qtext_counts = Counter(qtext for (_qid, qtext, _did, _dtext) in items)

            # Batch part (unique query_text)
            batch_input: Dict[str, List[str]] = {}
            batch_expected: Dict[str, List[str]] = {}
            unique_backmap: List[tuple[str, str, str]] = []  # (qid, query_text, doc_id)

            dup_items: List[tuple[str, str, str, str]] = []  # (qid, qtext, doc_id, doc_text)

            for qid, qtext, doc_id, doc_text in items:
                if qtext_counts[qtext] == 1:
                    batch_input[qtext] = [doc_text]
                    batch_expected[qtext] = expected_outputs[qid]
                    unique_backmap.append((qid, qtext, doc_id))
                else:
                    dup_items.append((qid, qtext, doc_id, doc_text))

            if batch_input:
                batch_generated = text_generator(batch_input)
                if set(batch_generated.keys()) != set(batch_input.keys()):
                    raise RuntimeError(
                        "The text_generator function did not return outputs for all given inputs."
                    )

                batch_scores = downstream_metric(batch_generated, batch_expected)
                if set(batch_scores.keys()) != set(batch_generated.keys()):
                    raise RuntimeError(
                        "The downstream_metric function did not return evaluation scores for all given inputs."
                    )

                for qid, qtext, doc_id in unique_backmap:
                    evaluation_scores[qid][doc_id] = batch_scores[qtext]

            # Duplicate query_text in the same batch: handle one-by-one (safe + simple)
            for qid, qtext, doc_id, doc_text in dup_items:
                generated = text_generator({qtext: [doc_text]})
                if qtext not in generated:
                    raise RuntimeError(
                        "The text_generator function did not return outputs for all given inputs."
                    )
                scores = downstream_metric(generated, {qtext: expected_outputs[qid]})
                if qtext not in scores:
                    raise RuntimeError(
                        "The downstream_metric function did not return evaluation scores for all given inputs."
                    )
                evaluation_scores[qid][doc_id] = scores[qtext]

    else:
        # Legacy mode: retrieval_results is {query_text: [doc_text, ...]}
        max_length_retrieval_lists = max((len(lst) for lst in retrieval_results.values()), default=0)

        flatten_inputs = {
            f"{query}@{i}": {"query": query, "document": [doc]}
            for query, documents in retrieval_results.items()
            for i, doc in enumerate(documents)
        }

        evaluation_scores_flat: Dict[str, Union[int, float]] = {}

        for i in range(max_length_retrieval_lists):
            current_input = {}
            current_expected_outputs = {}
            for query in retrieval_results.keys():
                key = f"{query}@{i}"
                if key in flatten_inputs:
                    item = flatten_inputs[key]
                    current_input[item["query"]] = item["document"]
                    current_expected_outputs[item["query"]] = expected_outputs[item["query"]]

            if not current_input:
                continue

            current_generated_outputs = text_generator(current_input)
            assert set(current_input.keys()) == set(
                current_generated_outputs.keys()
            ), "The text_generator function did not return outputs for all given inputs."

            current_evaluation_scores = downstream_metric(current_generated_outputs, current_expected_outputs)
            assert set(current_generated_outputs.keys()) == set(
                current_evaluation_scores.keys()
            ), "The downstream_metric function did not return evaluation scores for all given inputs."

            for query, score in current_evaluation_scores.items():
                evaluation_scores_flat[f"{query}@{i}"] = score

        # Convert to evaluation_scores[query_text][rank_as_docid]
        evaluation_scores = {q: {} for q in retrieval_results.keys()}
        for q in retrieval_results.keys():
            for j in range(len(retrieval_results[q])):
                evaluation_scores[q][str(j)] = evaluation_scores_flat.get(f"{q}@{j}", 0.0)

    # -----------------------------
    # Build qrel/run
    # -----------------------------
    qrel: Dict[str, Dict[str, Union[int, float]]] = {}
    run: Dict[str, Dict[str, float]] = {}
    binary_downstream_metric = True

    for qkey, docs in retrieval_results.items():
        run[qkey] = {}
        qrel[qkey] = {}

        for rank, doc in enumerate(docs):
            if inputs_are_ids:
                doc_key = str(doc)  # real doc_id
                score_val = evaluation_scores[qkey].get(doc_key, 0.0)
            else:
                doc_key = str(rank)  # legacy: rank as doc_id
                score_val = evaluation_scores[qkey].get(doc_key, 0.0)

            # run score only to preserve ranking (higher is better)
            run[qkey][doc_key] = float(len(docs) - rank)
            qrel[qkey][doc_key] = score_val

            # binary/continuous checks
            if qrel[qkey][doc_key] in [0, 1]:
                qrel[qkey][doc_key] = int(qrel[qkey][doc_key])  # type: ignore[assignment]
            if qrel[qkey][doc_key] not in [0, 1]:
                binary_downstream_metric = False
            if float(qrel[qkey][doc_key]) > 1 or float(qrel[qkey][doc_key]) < 0:
                raise RuntimeError("The returning value of the downstream_metric must be in range [0,1].")

    # -----------------------------
    # Compute IR metrics
    # -----------------------------
    def _is_recip_rank_k(m: str) -> bool:
        m_low = m.lower()
        return m_low.startswith("recip_rank_") and len(m_low.split("cut_")) == 2 and m_low.split("cut_")[1].isdigit()

    def _top_k_run(run_dict: Dict[str, Dict[str, float]], k: int) -> Dict[str, Dict[str, float]]:
        # Works only when doc ids are numeric ranks (legacy). In ID-mode we still have the rank order
        # encoded in run scores, but doc ids are not numeric. We thus top-k by run score.
        out: Dict[str, Dict[str, float]] = {}
        for qid, docs in run_dict.items():
            # take top-k by score desc
            top = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]
            out[qid] = {d: s for d, s in top}
        return out

    recip_rank_metrics = {m for m in retrieval_metrics if _is_recip_rank_k(m)}
    pytrec_metrics = set(retrieval_metrics) - recip_rank_metrics

    if binary_downstream_metric:
        results = {q: {} for q in retrieval_results.keys()}

        if pytrec_metrics:
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_metrics)
            pytrec_results = evaluator.evaluate(run)
            for qid in results.keys():
                results[qid].update(pytrec_results.get(qid, {}))

        if recip_rank_metrics:
            rr_eval = pytrec_eval.RelevanceEvaluator(qrel, {"recip_rank"})
            ks = sorted({int(m.split("cut_")[1]) for m in recip_rank_metrics})
            rr_by_k = {}
            for k in ks:
                run_k = _top_k_run(run, k)
                rr_by_k[k] = rr_eval.evaluate(run_k)  # qid -> {"recip_rank": val}

            for m in recip_rank_metrics:
                k = int(m.split("cut_")[1])
                per_q = rr_by_k[k]
                for qid in results.keys():
                    results[qid][m] = per_q.get(qid, {}).get("recip_rank", 0.0)

    else:
        results = {}
        for query, labels in qrel.items():
            results[query] = {}
            for metric in retrieval_metrics:
                if "_" in metric:
                    metric_without_cut = metric[: metric.find("_")]
                    if metric_without_cut not in {"success", "P"}:
                        raise RuntimeError(
                            'The provided retrieval metrics cannot be used with continuous downsream metric. '
                            'The supported retrieval metrics are ["success", "P"]'
                        )
                    cut_value = int(metric[metric.find("_") + 1 :])
                else:
                    metric_without_cut = metric
                    cut_value = len(labels)

                # labels keys are doc_ids; we need a ranking order -> use run scores desc
                ranked_docids = [d for d, _s in sorted(run[query].items(), key=lambda x: x[1], reverse=True)]
                top_docids = ranked_docids[:cut_value]

                if metric_without_cut == "success":
                    max_value = 0.0
                    for d in top_docids:
                        max_value = max(max_value, float(labels.get(d, 0.0)))
                    results[query][metric] = max_value

                elif metric_without_cut == "P":
                    if cut_value <= 0:
                        results[query][metric] = 0.0
                    else:
                        mean_value = 0.0
                        for d in top_docids:
                            mean_value += float(labels.get(d, 0.0))
                        results[query][metric] = mean_value / float(cut_value)

                else:
                    raise RuntimeError(
                        'The provided retrieval metrics cannot be used with continuous downsream metric. '
                        'The supported retrieval metrics are ["success", "P"]'
                    )

    final_results = {"per_input": results, "aggregated": {}}

    for metric in retrieval_metrics:
        values = [value.get(metric, 0.0) for value in results.values()]
        final_results["aggregated"][metric] = (sum(values) / len(values)) if values else 0.0

    # -----------------------------
    # NEW: return all (query_id, doc_id, score) triplets (from downstream_metric)
    # -----------------------------
    triples: List[Dict[str, Union[str, int, float]]] = []
    for qid, docs in retrieval_results.items():
        for rank, doc in enumerate(docs):
            doc_id = str(doc) if inputs_are_ids else str(rank)
            score = qrel.get(qid, {}).get(doc_id, 0.0)
            triples.append({"query_id": str(qid), "doc_id": str(doc_id), "score": float(score)})

    final_results["triples"] = triples
    return final_results
import string
from collections import Counter


def normalize_answer(s):
    """Converts text to lowercase, removes punctuation and extra spaces."""
    return ' '.join(''.join(ch for ch in s.lower() if ch not in string.punctuation).split())

def exact_match_metric(generated_outputs, expected_outputs):
    """Computes if the generated text (normalized) exactly matches one of the gold answers (normalized).
    Returns a dict {query: score}, where score is 1 or 0."""
    return {query: 1 if any(normalize_answer(gen) == normalize_answer(gold) for gold in expected_outputs.get(query, [])) else 0 for query, gen in generated_outputs.items()}

def _f1_score(prediction, ground_truth):
    """Helper function to compute F1 score for a single prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_metric(generated_outputs, expected_outputs):
    """
    Computes the F1 score for each query.
    For each query, it takes the maximum F1 score over all possible gold answers.
    Returns a dict {query: f1_score}.
    """
    f1_scores = {}
    for query, gen_answer in generated_outputs.items():
        gold_answers = expected_outputs.get(query, [])
        if not gold_answers:
            f1_scores[query] = 0.0
            continue
        max_f1 = max(_f1_score(gen_answer, gold) for gold in gold_answers)
        f1_scores[query] = max_f1
    return f1_scores
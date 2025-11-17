from itertools import combinations
import time


def apriori(transactions, min_support):
    """
    Apriori frequent itemset mining using horizontal data format.

    transactions: list[set[str]]
    min_support: float in [0, 1]

    Returns:
      frequent_itemsets: dict[frozenset[str], float] (support)
    """
    n = len(transactions)
    if n == 0:
        return {}

    min_support_count = max(1, int(min_support * n + 1e-9))

    # 1-itemsets
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1

    L = {}
    Lk = {frozenset([i]): c for i, c in item_counts.items() if c >= min_support_count}
    L.update(Lk)
    k = 2

    while Lk:
        prev_itemsets = list(Lk.keys())
        candidates = set()

        # Candidate generation (join step)
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                union = prev_itemsets[i] | prev_itemsets[j]
                if len(union) == k:
                    # All (k-1)-subsets must be frequent (Apriori property)
                    all_subsets_frequent = True
                    for subset in combinations(union, k - 1):
                        if frozenset(subset) not in Lk:
                            all_subsets_frequent = False
                            break
                    if all_subsets_frequent:
                        candidates.add(union)

        # Count support for candidates
        counts = {c: 0 for c in candidates}
        for t in transactions:
            for c in candidates:
                if c.issubset(t):
                    counts[c] += 1

        # Filter by minimum support
        Lk = {c: cnt for c, cnt in counts.items() if cnt >= min_support_count}
        L.update(Lk)
        k += 1

    # Convert counts to relative support
    frequent_itemsets = {itemset: count / n for itemset, count in L.items()}
    return frequent_itemsets


def generate_association_rules(freq_itemsets, min_confidence):
    """
    Generate association rules from frequent itemsets.

    freq_itemsets: dict[frozenset[str], float] with support
    min_confidence: float in [0, 1]

    Returns:
      list of dict with keys:
        antecedent, consequent, support, confidence, lift
    """
    rules = []
    for itemset, supp_XY in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                if not consequent:
                    continue
                supp_X = freq_itemsets.get(antecedent)
                supp_Y = freq_itemsets.get(consequent)
                if supp_X is None or supp_Y is None or supp_X == 0:
                    continue
                confidence = supp_XY / supp_X
                if confidence < min_confidence:
                    continue
                lift = supp_XY / (supp_X * supp_Y) if supp_X * supp_Y > 0 else 0.0
                rules.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": supp_XY,
                    "confidence": confidence,
                    "lift": lift
                })
    return rules


def run_apriori_with_timing(transactions, min_support, min_confidence):
    """
    Run Apriori + rule generation and measure execution time.
    """
    start = time.perf_counter()
    freq = apriori(transactions, min_support)
    rules = generate_association_rules(freq, min_confidence)
    end = time.perf_counter()
    return {
        "frequent_itemsets": freq,
        "rules": rules,
        "time_ms": (end - start) * 1000.0
    }

import time
from itertools import combinations


def eclat(transactions, min_support):
    """
    Eclat frequent itemset mining using vertical TID-sets.

    transactions: list[set[str]]
    min_support: float in [0, 1]

    Returns:
      frequent_itemsets: dict[frozenset[str], float] (support)
    """
    n = len(transactions)
    if n == 0:
        return {}

    min_support_count = max(1, int(min_support * n + 1e-9))

    # Build vertical representation: item -> set of transaction IDs
    tidsets = {}
    for tid, t in enumerate(transactions):
        for item in t:
            tidsets.setdefault(frozenset([item]), set()).add(tid)

    # Keep only frequent singletons
    frequent_singletons = {
        itemset: tids for itemset, tids in tidsets.items()
        if len(tids) >= min_support_count
    }

    results = {}

    def dfs(prefix, prefix_tids, items):
        """
        Depth-first search over itemsets.
        prefix: frozenset
        prefix_tids: set of transaction IDs
        items: list of (itemset, tids)
        """
        for i in range(len(items)):
            itemset_i, tids_i = items[i]
            if prefix:
                new_itemset = prefix | itemset_i
                new_tids = prefix_tids & tids_i
            else:
                new_itemset = itemset_i
                new_tids = tids_i

            if len(new_tids) >= min_support_count:
                results[new_itemset] = new_tids
                dfs(new_itemset, new_tids, items[i + 1:])

    # Sort items for deterministic traversal
    singleton_items = sorted(frequent_singletons.items(),
                             key=lambda x: list(x[0])[0])

    dfs(frozenset(), set(range(n)), singleton_items)

    frequent_itemsets = {itemset: len(tids) / n for itemset, tids in results.items()}
    return frequent_itemsets


def generate_association_rules(freq_itemsets, min_confidence):
    """
    Same structure as Apriori rule generation, reused here for Eclat.
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


def run_eclat_with_timing(transactions, min_support, min_confidence):
    """
    Run Eclat + rule generation and measure execution time.
    """
    start = time.perf_counter()
    freq = eclat(transactions, min_support)
    rules = generate_association_rules(freq, min_confidence)
    end = time.perf_counter()
    return {
        "frequent_itemsets": freq,
        "rules": rules,
        "time_ms": (end - start) * 1000.0
    }

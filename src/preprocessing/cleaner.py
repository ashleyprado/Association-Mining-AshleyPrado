import pandas as pd


def load_products(path_or_file):
    """
    Load products.csv and return:
      - valid_products: set of standardized product names (lowercase, stripped)
      - df: original dataframe
    Assumes columns: product_id, product_name, category
    """
    df = pd.read_csv(path_or_file)
    df["product_name"] = df["product_name"].astype(str).str.strip().str.lower()
    valid_products = set(df["product_name"])
    return valid_products, df


def parse_transactions_df(df):
    """
    Convert a transaction dataframe into a list of transactions (list[list[str]]).
    Two possible formats:
      1) Column 'items' with comma-separated strings.
      2) Otherwise, assume first column is transaction id, others are item columns.
    """
    transactions = []

    if "items" in df.columns:
        for _, row in df.iterrows():
            raw = str(row["items"])
            if raw.lower() in ("nan", "none"):
                transactions.append([])
            else:
                items = [p.strip() for p in raw.split(",") if p.strip() != ""]
                transactions.append(items)
    else:
        item_cols = df.columns[1:]
        for _, row in df.iterrows():
            items = []
            for col in item_cols:
                val = str(row[col]).strip()
                if val and val.lower() not in ("nan", "none"):
                    items.append(val)
            transactions.append(items)

    return transactions


def preprocess_transactions(raw_transactions, valid_products):
    """
    Apply all required cleaning steps:
      - Empty transactions removal
      - Single-item transactions removal
      - Duplicate items removal within a transaction
      - Product name standardization (lowercase, stripped)
      - Invalid product removal (not in valid_products)

    Returns:
      cleaned_transactions: list[set[str]]
      report: dict with statistics for the report
    """
    report = {
        "total_transactions_before": len(raw_transactions),
        "empty_transactions": 0,
        "single_item_transactions": 0,
        "duplicate_items_instances": 0,
        "invalid_items_instances": 0,
    }

    cleaned_transactions = []

    for items in raw_transactions:
        standardized = []
        seen = set()

        for item in items:
            item_clean = str(item).strip().lower()
            if not item_clean:
                continue

            # Invalid product
            if item_clean not in valid_products:
                report["invalid_items_instances"] += 1
                continue

            # Duplicate within transaction
            if item_clean in seen:
                report["duplicate_items_instances"] += 1
                continue

            seen.add(item_clean)
            standardized.append(item_clean)

        # Empty after cleaning
        if len(standardized) == 0:
            report["empty_transactions"] += 1
            continue

        # Single-item transactions are removed
        if len(standardized) == 1:
            report["single_item_transactions"] += 1
            continue

        cleaned_transactions.append(set(standardized))

    report["total_transactions_after"] = len(cleaned_transactions)
    report["total_items_after"] = sum(len(t) for t in cleaned_transactions)
    all_items = set().union(*cleaned_transactions) if cleaned_transactions else set()
    report["unique_products_after"] = len(all_items)

    return cleaned_transactions, report

import pandas as pd
import streamlit as st

from preprocessing.cleaner import (
    load_products,
    parse_transactions_df,
    preprocess_transactions,
)
from algorithms.apriori import run_apriori_with_timing
from algorithms.eclat import run_eclat_with_timing


# ----------------- Streamlit Page Config -----------------

st.set_page_config(page_title="Supermarket Association Rule Mining", layout="wide")

# ----------------- Session State Initialization -----------------

if "current_transaction" not in st.session_state:
    st.session_state.current_transaction = []

if "manual_transactions" not in st.session_state:
    st.session_state.manual_transactions = []

if "cleaned_transactions" not in st.session_state:
    st.session_state.cleaned_transactions = []

if "preprocess_report" not in st.session_state:
    st.session_state.preprocess_report = None

if "apriori_result" not in st.session_state:
    st.session_state.apriori_result = None

if "eclat_result" not in st.session_state:
    st.session_state.eclat_result = None

if "all_products_list" not in st.session_state:
    st.session_state.all_products_list = []


# ----------------- Sidebar Parameters -----------------

st.sidebar.header("Mining Parameters")
min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.2, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.01, 1.0, 0.5, 0.01)

st.sidebar.info("Adjust thresholds, then run Apriori and Eclat in the main panel.")


# ----------------- Title -----------------

st.title("Interactive Supermarket Simulation with Association Rule Mining")

# ----------------- Step 1: Load Product List -----------------

st.subheader("Step 1. Load Product List (products.csv)")

products_file = st.file_uploader("Upload products.csv (optional)", type="csv")

if products_file is not None:
    # Use uploaded file
    valid_products, products_df = load_products(products_file)
else:
    # Fall back to default path
    try:
        valid_products, products_df = load_products("data/products.csv")
        st.caption("Loaded default data/products.csv")
    except Exception:
        valid_products, products_df = set(), None
        st.error("Could not load products.csv. Please upload it above.")

if valid_products:
    st.success(f"Loaded {len(valid_products)} valid products.")
    st.session_state.all_products_list = sorted(valid_products)
    if st.checkbox("Show product list"):
        st.dataframe(products_df)
else:
    st.info("No valid products loaded yet. Manual transaction creation will be disabled.")


# ----------------- Step 2: Manual Transaction Creation -----------------

st.subheader("Step 2. Create Manual Transactions")

if st.session_state.all_products_list:
    st.write("Click products to add them to the current transaction:")

    product_list = st.session_state.all_products_list
    cols = st.columns(5)
    for idx, prod in enumerate(product_list):
        col = cols[idx % 5]
        if col.button(prod, key=f"prod_{prod}"):
            st.session_state.current_transaction.append(prod)

    st.write("Current transaction:", st.session_state.current_transaction)

    col1, col2 = st.columns(2)
    if col1.button("Save transaction"):
        if st.session_state.current_transaction:
            st.session_state.manual_transactions.append(
                list(st.session_state.current_transaction)
            )
            st.session_state.current_transaction = []
            st.success("Transaction saved.")
        else:
            st.warning("No items to save. Add at least one product.")

    if col2.button("Clear current transaction"):
        st.session_state.current_transaction = []

    if st.session_state.manual_transactions:
        st.write("Manual transactions created so far:")
        df_manual = pd.DataFrame({
            "Transaction #": range(1, len(st.session_state.manual_transactions) + 1),
            "Items": [", ".join(t) for t in st.session_state.manual_transactions]
        })
        st.dataframe(df_manual)
else:
    st.info("Load products first to enable manual transaction creation.")


# ----------------- Step 3: CSV Import -----------------

st.subheader("Step 3. Import Transactions from CSV (sample_transactions.csv)")

transactions_file = st.file_uploader("Upload sample_transactions.csv (optional)", type="csv")

raw_transactions = []

if transactions_file is not None:
    try:
        df_tx = pd.read_csv(transactions_file)
        st.info(f"Loaded {len(df_tx)} transactions from uploaded sample_transactions.csv.")
        raw_transactions = parse_transactions_df(df_tx)
    except Exception as e:
        st.error(f"Error reading uploaded transactions CSV: {e}")
else:
    # Load default
    try:
        df_tx = pd.read_csv("data/sample_transactions.csv")
        st.caption("Loaded default data/sample_transactions.csv")
        st.info(f"Default file has {len(df_tx)} transactions.")
        raw_transactions = parse_transactions_df(df_tx)
    except Exception:
        st.warning("No transaction CSV loaded or found.")

# Add manual transactions
if st.session_state.manual_transactions:
    st.write(f"Adding {len(st.session_state.manual_transactions)} manual transactions.")
    raw_transactions.extend(st.session_state.manual_transactions)

st.write(f"Total raw transactions (CSV + manual): {len(raw_transactions)}")

if st.checkbox("Show raw transactions (before preprocessing)"):
    df_raw = pd.DataFrame({
        "Transaction #": range(1, len(raw_transactions) + 1),
        "Items": [", ".join(map(str, t)) for t in raw_transactions]
    })
    st.dataframe(df_raw)


# ----------------- Step 4: Preprocessing -----------------

st.subheader("Step 4. Data Preprocessing")

if st.button("Run preprocessing"):
    if not valid_products:
        st.error("Load products.csv first.")
    elif not raw_transactions:
        st.error("No transactions to preprocess.")
    else:
        cleaned, report = preprocess_transactions(raw_transactions, valid_products)
        st.session_state.cleaned_transactions = cleaned
        st.session_state.preprocess_report = report
        st.success("Preprocessing completed.")

if st.session_state.preprocess_report:
    rep = st.session_state.preprocess_report

    st.markdown("**Preprocessing Report**")
    st.write("Before Cleaning:")
    st.write(f"Total transactions: {rep['total_transactions_before']}")
    st.write(f"Empty transactions: {rep['empty_transactions']}")
    st.write(f"Single-item transactions: {rep['single_item_transactions']}")
    st.write(f"Duplicate items found: {rep['duplicate_items_instances']} instances")
    st.write(f"Invalid items found: {rep['invalid_items_instances']} instances")

    st.write("After Cleaning:")
    st.write(f"Valid transactions: {rep['total_transactions_after']}")
    st.write(f"Total items: {rep['total_items_after']}")
    st.write(f"Unique products: {rep['unique_products_after']}")

    if st.checkbox("Show cleaned transactions"):
        df_clean = pd.DataFrame({
            "Transaction #": range(1, len(st.session_state.cleaned_transactions) + 1),
            "Items": [", ".join(sorted(list(t))) for t in st.session_state.cleaned_transactions]
        })
        st.dataframe(df_clean)


# ----------------- Step 5: Run Apriori and Eclat -----------------

st.subheader("Step 5. Run Association Rule Mining")

if st.button("Run Apriori and Eclat"):
    tx = st.session_state.cleaned_transactions
    if not tx:
        st.error("No cleaned transactions. Run preprocessing first.")
    else:
        st.info("Running Apriori...")
        apriori_result = run_apriori_with_timing(tx, min_support, min_confidence)
        st.session_state.apriori_result = apriori_result
        st.success(f"Apriori generated {len(apriori_result['rules'])} rules.")

        st.info("Running Eclat...")
        eclat_result = run_eclat_with_timing(tx, min_support, min_confidence)
        st.session_state.eclat_result = eclat_result
        st.success(f"Eclat generated {len(eclat_result['rules'])} rules.")

if st.session_state.apriori_result and st.session_state.eclat_result:
    st.markdown("### Performance Comparison")
    comp_df = pd.DataFrame([
        {
            "Algorithm": "Apriori",
            "Execution time (ms)": round(st.session_state.apriori_result["time_ms"], 2),
            "Number of rules": len(st.session_state.apriori_result["rules"])
        },
        {
            "Algorithm": "Eclat",
            "Execution time (ms)": round(st.session_state.eclat_result["time_ms"], 2),
            "Number of rules": len(st.session_state.eclat_result["rules"])
        }
    ])
    st.table(comp_df)
    st.bar_chart(comp_df.set_index("Algorithm")[["Execution time (ms)"]])


# ----------------- Step 6: Recommendation UI -----------------

st.subheader("Step 6. Product Recommendation (User-Friendly Output)")


def strength_label(conf):
    if conf >= 0.7:
        return "Strong"
    elif conf >= 0.5:
        return "Moderate"
    else:
        return "Weak"


all_items = sorted({
    item for t in st.session_state.cleaned_transactions for item in t
}) if st.session_state.cleaned_transactions else []

if all_items:
    selected_product = st.selectbox("Select a product to query:", all_items)
    algo_choice = st.radio("Algorithm for recommendations:", ["Apriori", "Eclat"])

    rules_source = None
    if algo_choice == "Apriori" and st.session_state.apriori_result:
        rules_source = st.session_state.apriori_result["rules"]
    elif algo_choice == "Eclat" and st.session_state.eclat_result:
        rules_source = st.session_state.eclat_result["rules"]
    else:
        st.info("Run the selected algorithm in Step 5 first.")

    if rules_source:
        related_rules = [
            r for r in rules_source if selected_product in r["antecedent"]
        ]
        related_rules.sort(key=lambda r: r["confidence"], reverse=True)

        if related_rules:
            st.markdown(f"**Customers who bought `{selected_product}` also bought:**")
            for r in related_rules[:10]:
                consequent_items = ", ".join(sorted(list(r["consequent"])))
                conf_pct = r["confidence"] * 100
                st.write(
                    f"- {consequent_items}: {conf_pct:.1f}% of the time "
                    f"({strength_label(r['confidence'])})"
                )
                st.progress(min(1.0, r["confidence"]))

            st.markdown("**Business Recommendation:**")
            st.write(
                f"Consider placing `{selected_product}` near its most frequently associated products "
                f"or offering bundles that include them."
            )
        else:
            st.write("No association rules found for this product with the current thresholds.")
else:
    st.info("Preprocess data and run mining to enable recommendations.")

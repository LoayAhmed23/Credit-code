import pandas as pd
import numpy as np
import glob
import os


# ========================= Helper Functions (from main.py) =========================
def apply_cast_and_report(df, columns, cast_type):
    # Get the total number of rows for our percentage math
    total_rows = len(df)
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue
            
        # Count nulls before
        nulls_before = df[col].isna().sum()
        
        # Apply the specific vectorized casting logic
        if cast_type == 'string':
            df[col] = df[col].astype("string")
            
        elif cast_type == 'float':
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        elif cast_type == 'int':
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            
        elif cast_type == 'date':
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        # Count nulls after
        nulls_after = df[col].isna().sum()
        
        # Calculate percentage of missing data
        pct_missing = (nulls_after / total_rows) * 100
        
        # Print the results, adding the percentage formatted to 2 decimal places
        print(f"[{col}] Type: {df[col].dtype} | Nulls: {nulls_before} -> {nulls_after} ({pct_missing:.2f}% missing)")


# ========================= Column Definitions (from main.py) =========================
prime_string_cols = [
    "BRANCH_NAME", "ACTIVATED", "STATUS", "STATUS_NAME",
    "PRODUCT_NAME", "GENDER", "CUSTOMER_TYPE", "Card account status "
]
prime_int_cols = ["BRANCH_ID", "RIMNO"]
prime_float_cols = [
    "CREDIT_LIMIT", "DELIQUENCY", "LEDGER_BALANCE", "AVAILABLE_LIMIT",
    "OVERDUEAMOUNT", "FIRST_REPLACED_CARD", "SECOND_REPLACED_CARD",
    "THIRD_REPLACED_CARD"
]
prime_date_cols = ["CREATION_DATE", "LAST_STATEMENT_DATE", "DOB", "CLOSURE_DATE"]


# ========================= Output Directory =========================
output_dir = "prime_cleaned"
os.makedirs(output_dir, exist_ok=True)


# ========================= Find All Prime Files =========================
prime_files = glob.glob("prime/*.csv")

if not prime_files:
    print("ERROR: No CSV files found in 'prime/' folder. Make sure the folder exists.")
    exit(1)

print(f"Found {len(prime_files)} prime file(s).\n")

# Status codes that indicate an inactive / closed / blocked card
inactive_statuses = [
    'CLSB',   # closed by bank
    'CLSC',   # closed by customer
    'LOST',   # lost card
    'WROF',   # write off status
    'CNCD',   # card not collected by DSU
    'SUSP',   # profit suspended
    'FRAD',   # pick up card special fraud
    'CLSD',   # closed
    'BLOK',   # blok card
    'NOAU',   # no authorization
    'EXMU',   # expired murabha
    'PICK',   # pick up card
    'BLCK',   # blocked status
    'STLC',   # stolen card
    'OFBL',   # offline pin block
    'ONBL',   # online pin block
    'FREZ',   # freeze card
    'EXPD',   # expired
    'EXPC',   # expired card
]


# ========================= PASS 1: Build Global CUSTOMER_ID Mapping =========================
print("PASS 1: Scanning all files to build a global CUSTOMER_ID mapping...\n")

all_pairs = []

for file in prime_files:
    print(f"  Scanning: {file}")
    temp_df = pd.read_csv(
        file,
        encoding='latin',
        usecols=['RIMNO', 'RIMNO', 'DOB'],  # only need the key columns
        dtype='string'
    )

    # Standardize column name
    if 'RIMNO' in temp_df.columns:
        temp_df = temp_df.rename(columns={'RIMNO': 'RIMNO'})

    temp_df['RIMNO'] = temp_df['RIMNO'].str.strip()
    temp_df['DOB'] = pd.to_datetime(temp_df['DOB'], errors='coerce')

    all_pairs.append(temp_df[['RIMNO', 'DOB']])

# Combine and deduplicate across ALL files
global_pairs = pd.concat(all_pairs, ignore_index=True).drop_duplicates().reset_index(drop=True)
global_pairs['CUSTOMER_ID'] = range(1, len(global_pairs) + 1)

print(f"\n  Global unique (RIMNO, DOB) pairs: {len(global_pairs)}")
print(f"  CUSTOMER_ID range: 1 - {len(global_pairs)}\n")


# ========================= PASS 2: Process Each File Separately =========================
print("PASS 2: Processing each file with the global CUSTOMER_ID mapping...\n")

for file in prime_files:
    file_basename = os.path.splitext(os.path.basename(file))[0]
    print(f"{'='*60}")
    print(f"Processing: {file}")
    print(f"{'='*60}")

    # --- Load ---
    df = pd.read_csv(
        file,
        encoding='latin',
        dtype={col: "string" for col in prime_string_cols + prime_int_cols + prime_float_cols},
        parse_dates=prime_date_cols
    ).rename(columns={'RIMNO': 'RIMNO', 'NAME': 'PRODUCT_NAME'})

    print(f"  Rows loaded: {len(df)}")

    # --- Cast all columns using helper function ---
    print("\n  -> String Columns:")
    apply_cast_and_report(df, prime_string_cols, 'string')

    print("\n  -> Float Columns:")
    apply_cast_and_report(df, prime_float_cols, 'float')

    print("\n  -> Integer Columns:")
    apply_cast_and_report(df, prime_int_cols, 'int')

    print("\n  -> Date Columns:")
    apply_cast_and_report(df, prime_date_cols, 'date')

    # Clean up ACTIVATED and STATUS for reliable splitting
    df['STATUS'] = df['STATUS'].astype("string").str.strip().str.upper()
    df['Card account status '] = df['Card account status '].astype("string").str.strip().str.upper()

    # --- Assign CUSTOMER_ID from the global mapping ---
    # Prepare RIMNO to match the format used in the global mapping
    df['RIMNO'] = df['RIMNO'].astype("string").str.strip()
    df = df.merge(global_pairs, on=['RIMNO', 'DOB'], how='left')

    assigned = df['CUSTOMER_ID'].notna().sum()
    unmatched = df['CUSTOMER_ID'].isna().sum()
    print(f"\n  CUSTOMER_ID assigned: {assigned} rows")
    if unmatched > 0:
        print(f"  WARNING: {unmatched} rows could not be assigned a CUSTOMER_ID (missing RIMNO or DOB).")

    # --- Split into Active & Historical ---
    # Historical: ACTIVATED is D/I  OR  STATUS is an inactive code
    is_inactive_status = df['STATUS'].isin(inactive_statuses)
    is_inactive_card = df['Card account status '].isin(inactive_statuses)
    is_historical = is_inactive_status & is_inactive_card

    historical_df = df[is_historical].copy()
    active_df = df[~is_historical].copy()

    print(f"  Active   (ACTIVATED='A' & active STATUS):   {len(active_df)} rows")
    print(f"  Historical (ACTIVATED='D'/'I' or inactive STATUS): {len(historical_df)} rows")

    # --- Detect relatives: same (RIMNO, PRODUCT_NAME) but different CUSTOMER_ID ---
    # For each (RIMNO, PRODUCT_NAME) group, keep only the row(s) with the oldest DOB
    # and move the rest to active_relatives
    dup_check = active_df.groupby(['RIMNO', 'PRODUCT_NAME'])['CUSTOMER_ID'].nunique()
    dup_groups = dup_check[dup_check > 1].index  # (RIMNO, PRODUCT_NAME) pairs with >1 CUSTOMER_ID

    if len(dup_groups) > 0:
        print(f"\n  Found {len(dup_groups)} (RIMNO, PRODUCT_NAME) pair(s) with multiple CUSTOMER_IDs.")

        # For each duplicated group, find the oldest DOB (min DOB = oldest person)
        oldest_dob = (
            active_df[active_df.set_index(['RIMNO', 'PRODUCT_NAME']).index.isin(dup_groups)]
            .groupby(['RIMNO', 'PRODUCT_NAME'])['DOB']
            .min()
            .reset_index()
            .rename(columns={'DOB': 'OLDEST_DOB'})
        )

        # Merge to tag each row
        active_df = active_df.merge(oldest_dob, on=['RIMNO', 'PRODUCT_NAME'], how='left')

        # Rows in duplicated groups whose DOB is NOT the oldest -> relatives
        is_dup_group = active_df.set_index(['RIMNO', 'PRODUCT_NAME']).index.isin(dup_groups)
        is_not_oldest = active_df['DOB'] != active_df['OLDEST_DOB']

        relatives_df = active_df[is_dup_group & is_not_oldest].copy()
        active_df = active_df[~(is_dup_group & is_not_oldest)].copy()

        # Clean up helper column
        active_df = active_df.drop(columns=['OLDEST_DOB'])
        relatives_df = relatives_df.drop(columns=['OLDEST_DOB'])

        print(f"  Moved {len(relatives_df)} row(s) to active_relatives.")
        print(f"  Active after dedup: {len(active_df)} rows")
    else:
        relatives_df = pd.DataFrame()
        print(f"\n  No (RIMNO, PRODUCT_NAME) duplicates found — all CUSTOMER_IDs are unique per combo.")

    # --- Save ---
    active_path = os.path.join(output_dir, f"{file_basename}_active.csv")
    historical_path = os.path.join(output_dir, f"{file_basename}_historical.csv")

    active_df.to_csv(active_path, index=False)
    historical_df.to_csv(historical_path, index=False)

    print(f"  Saved -> {active_path}")
    print(f"  Saved -> {historical_path}")

    if len(relatives_df) > 0:
        relatives_path = os.path.join(output_dir, f"{file_basename}_active_relatives.csv")
        relatives_df.to_csv(relatives_path, index=False)
        print(f"  Saved -> {relatives_path}")

    print()

print("All files processed!")

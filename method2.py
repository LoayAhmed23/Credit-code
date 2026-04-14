"""
add_churn.py
------------
Reads each prime CSV independently, derives a CHURN flag per RIMNO
within that file, then concatenates all labelled files at the end.

Churn rule (applied per file)
------------------------------
  CHURN = 1  for a RIMNO  ←→  every STATUS for that RIMNO in THIS file
                               is a churn status.
  CHURN = 0               ←→  the RIMNO has at least one non-churn STATUS
                               in this file.
"""

import glob   # used to find all files matching a pattern (e.g. all *.csv files in a folder)
import os     # used to build file paths and create folders

import pandas as pd  # the main library for working with tabular data (like Excel but in Python)


# ---------------------------------------------------------------------------
# CONFIGURE ME: statuses that mean the customer has permanently left
# ---------------------------------------------------------------------------
# A Python "set" — like a list but faster for checking "is this value in here?"
# We use .upper() everywhere so matching is case-insensitive (clsb == CLSB)
CHURN_STATUSES = {
    "CLSB",   # closed by bank
    "CLSC",   # closed by customer
    "CLSD",   # closed
    "WROF",   # written off
    "EXPD",   # expired
    "EXPC",   # expired card
}

# ---------------------------------------------------------------------------
# Paths — change these if your folders are named differently
# ---------------------------------------------------------------------------
PRIME_DIR  = "prime_data"                        # folder where your CSVs live
OUTPUT_DIR = os.path.join(PRIME_DIR, "with_churn")  # subfolder we'll write results to
                                                     # os.path.join builds "prime_data/with_churn"
RIMNO_COL  = "RIM_NO"    # the customer ID column name as it appears in the raw CSV
STATUS_COL = "STATUS"    # the status column name


# ---------------------------------------------------------------------------
# Find all CSV files in the prime_data folder
# ---------------------------------------------------------------------------
# glob.glob() searches for files matching a pattern and returns a list of paths
# os.path.join(PRIME_DIR, "*.csv") builds the pattern "prime_data/*.csv"
# sorted() puts them in alphabetical order so processing is predictable
files = sorted(glob.glob(os.path.join(PRIME_DIR, "*.csv")))

# If no files were found, stop immediately with a clear error message
if not files:
    raise FileNotFoundError(f"No CSV files found in '{PRIME_DIR}'")

# Create the output folder if it doesn't already exist
# exist_ok=True means "don't crash if the folder is already there"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# This list will collect the processed DataFrame from each file
# so we can concatenate them all at the end
all_frames = []


# ---------------------------------------------------------------------------
# Process each file independently
# ---------------------------------------------------------------------------
# We loop through every file path one at a time
for path in files:

    # pd.read_csv() reads a CSV file into a DataFrame (think: an in-memory table)
    # encoding="latin" handles special characters that can appear in Arabic/French bank data
    # dtype=str means "read every column as plain text" — this prevents pandas from
    #   misreading things like "0090" as the number 90, or dates as numbers
    df = pd.read_csv(path, encoding="latin", dtype=str)

    # os.path.basename() strips the folder path and keeps just the filename
    # e.g. "prime_data/january.csv" → "january.csv"
    fname = os.path.basename(path)
    print(f"\nProcessing: {fname}  ({len(df):,} rows)")
    # len(df) = number of rows in this file


    # --- Step 1: Normalise the STATUS column ---
    # We create a new helper column "_status_norm" that is a clean version of STATUS
    # .astype(str)  → make sure every value is a string (handles NaN, numbers, etc.)
    # .str.strip()  → remove accidental leading/trailing spaces ("NORM " → "NORM")
    # .str.upper()  → convert to uppercase ("norm" → "NORM") so matching is consistent
    df["_status_norm"] = df[STATUS_COL].astype(str).str.strip().str.upper()


    # --- Step 2: Flag each ROW as churn or not ---
    # .isin(CHURN_STATUSES) checks if each value in "_status_norm" exists in our set
    # It returns True/False for every row
    # We store this as a new helper column "_is_churn_status"
    # Example: if STATUS = "CLSB" → True | if STATUS = "NORM" → False
    df["_is_churn_status"] = df["_status_norm"].isin(CHURN_STATUSES)


    # --- Step 3: Decide churn at the CUSTOMER level (not the row level) ---
    # A single customer (RIMNO) can have multiple rows in one file.
    # We want CHURN=1 only if ALL of that customer's rows are churn statuses.
    # If even one row is non-churn (e.g. NORM), the customer is NOT churned.

    # .groupby(RIMNO_COL) → groups all rows that share the same RIM_NO together
    # ["_is_churn_status"] → look at this column within each group
    # .all() → returns True for a group only if EVERY value in it is True
    #          (i.e. every row for this customer is a churn status)
    # .rename("_all_churn") → give the result column a clear name
    # .reset_index() → turns the result back into a regular table with two columns:
    #                  RIM_NO and _all_churn
    rimno_all_churn = (
        df.groupby(RIMNO_COL)["_is_churn_status"]
        .all()
        .rename("_all_churn")
        .reset_index()
    )
    # At this point rimno_all_churn looks like:
    #   RIM_NO   | _all_churn
    #   12345    | True
    #   67890    | False


    # --- Step 4: Bring the customer-level decision back to the row level ---
    # df.merge() joins two tables together, like a VLOOKUP in Excel
    # on=RIMNO_COL → match rows using the RIM_NO column
    # how="left"   → keep all rows from df (the left table); if a RIMNO somehow
    #                isn't in rimno_all_churn, fill with NaN instead of dropping the row
    df = df.merge(rimno_all_churn, on=RIMNO_COL, how="left")
    # Now every row in df has an "_all_churn" column copied from the customer-level result


    # --- Step 5: Convert True/False to 1/0 ---
    # .astype(int) converts boolean True → 1, False → 0
    df["CHURN"] = df["_all_churn"].astype(int)


    # --- Step 6: Print a summary for this file so you can sanity-check ---
    # This groups by RIMNO, takes the first CHURN value per customer (they're all the same),
    # then counts how many unique customers got CHURN=0 vs CHURN=1
    churn_summary = (
        df.groupby(RIMNO_COL)["CHURN"]
        .first()           # one value per customer (all rows for a customer are identical)
        .value_counts()    # count how many customers have CHURN=0 and how many have CHURN=1
        .rename_axis("CHURN")
        .reset_index(name="unique_customers")
    )
    print(f"  Churn distribution (unique customers in {fname}):")
    print(churn_summary.to_string(index=False))


    # --- Step 7: Clean up the helper columns we no longer need ---
    # These were only needed to compute CHURN; we don't want them in the output file
    df = df.drop(columns=["_status_norm", "_is_churn_status", "_all_churn"])


    # --- Step 8: Write the updated file to the output folder ---
    # os.path.join builds "prime_data/with_churn/january.csv"
    out_path = os.path.join(OUTPUT_DIR, fname)
    # to_csv() writes the DataFrame back to a CSV file
    # index=False means "don't write the row numbers (0,1,2,...) as an extra column"
    df.to_csv(out_path, index=False)
    print(f"  Wrote -> {out_path}")


    # --- Step 9: Save this processed DataFrame for later concatenation ---
    all_frames.append(df)
    # .append() adds the DataFrame to the end of the list


# ---------------------------------------------------------------------------
# Concatenate all labelled files into one big DataFrame
# ---------------------------------------------------------------------------
# pd.concat() stacks a list of DataFrames on top of each other (like stacking Excel sheets)
# ignore_index=True resets the row numbers so they run 0,1,2,... continuously
#   instead of 0..N from file1 then 0..M from file2
combined = pd.concat(all_frames, ignore_index=True)

print(f"\nCombined shape: {combined.shape}")
# .shape returns (number_of_rows, number_of_columns) as a tuple

print("\nDone. Updated files are in:", OUTPUT_DIR)
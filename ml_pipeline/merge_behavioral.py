import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
enriched_path = os.path.join(BASE_DIR, "..", "data", "processed", "enriched_dataset.csv")
orig_path = os.path.join(BASE_DIR, "..", "data", "processed", "accepted_loans_final.csv")

print("Loading enriched dataset...")
df_enriched = pd.read_csv(enriched_path, low_memory=False)

print("Loading behavioral columns from original dataset...")
cols_to_fetch = [
    "loan_status", "term", "verification_status", "inq_last_6mths", "delinq_2yrs",
    "pub_rec", "installment", "acc_open_past_24mths", "mort_acc", "tot_cur_bal",
    "bc_util", "percent_bc_gt_75", "pub_rec_bankruptcies", "num_tl_op_past_12m",
    "tot_hi_cred_lim", "open_acc", "earliest_cr_line", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "mths_since_recent_bc", "mths_since_recent_inq", "mths_since_rcnt_il"
]

# Ensure we only fetch columns that actually exist in original and are NOT already in enriched
existing_orig = pd.read_csv(orig_path, nrows=0).columns.tolist()
valid_cols = [c for c in cols_to_fetch if c in existing_orig and c not in df_enriched.columns]

df_orig = pd.read_csv(orig_path, usecols=valid_cols, low_memory=False)

print("Concatenating datasets...")
# Since rows are exactly matched, we can concat horizontally
df_final = pd.concat([df_enriched, df_orig], axis=1)

print(f"Final shape: {df_final.shape}")
print("Saving back to enriched_dataset.csv...")
df_final.to_csv(enriched_path, index=False)

print("Merge completed successfully!")

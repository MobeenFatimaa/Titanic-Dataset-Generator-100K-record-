import pandas as pd
import numpy as np

FILE_NAME = "titanic_2_0_passenger_survival_dataset.csv"

def validate_titanic_dataset(file_path):
    print("=" * 60)
    print("TITANIC 2.0 DATASET VALIDATION REPORT")
    print("=" * 60)
    
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
        print(f"✓ File successfully loaded.")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return

    # 2. Key Target Verification
    print("-" * 60)
    print("1. TARGET VARIABLE VERIFICATION ('survived')")
    print("-" * 60)
    if 'survived' in df.columns:
        survival_rate = df['survived'].mean()
        class_counts = df['survived'].value_counts()
        print(f"Target Distribution:\n{class_counts.to_string()}")
        print(f"Overall Survival Rate: {survival_rate:.2%}")
        
        if 0.25 <= survival_rate <= 0.35:
            print("✓ PASS: Survival rate falls within target range (25% - 35%).")
        else:
            print("⚠ WARNING: Survival rate outside expected 25%-35% range.")
    else:
        print("✗ FAIL: Target column 'survived' not found.")

    # 3. Missing Value Analysis
    print("\n" + "-" * 60)
    print("2. MISSING VALUE ANALYSIS")
    print("-" * 60)
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"  Column '{col}': {count:,} missing values ({pct:.2f}%)")
    else:
        print("  No missing values found across all columns.")

    # 4. Logical & Feature Integrity Checks
    print("\n" + "-" * 60)
    print("3. LOGICAL & DATA INTEGRITY CHECKS")
    print("-" * 60)
    
    # Check ID Uniqueness
    unique_ids = df['passenger_id'].nunique()
    if unique_ids == len(df):
        print(f"✓ PASS: 'passenger_id' is 100% unique ({unique_ids:,} records).")
    else:
        print(f"✗ FAIL: Duplicate passenger IDs detected ({len(df) - unique_ids} duplicates).")
        
    # Check Family Size logic: family_size == siblings_spouses + parents_children + 1
    family_calc = df['siblings_spouses'] + df['parents_children'] + 1
    family_mismatch = (df['family_size'] != family_calc).sum()
    if family_mismatch == 0:
        print("✓ PASS: 'family_size' matches (siblings_spouses + parents_children + 1) across all rows.")
    else:
        print(f"✗ FAIL: Mismatch in 'family_size' calculation for {family_mismatch} rows.")
        
    # Check 'is_alone' flag
    alone_mismatch = ((df['family_size'] == 1) != (df['is_alone'] == 1)).sum()
    if alone_mismatch == 0:
        print("✓ PASS: 'is_alone' binary indicator correctly matches 'family_size'.")
    else:
        print(f"✗ FAIL: Mismatch in 'is_alone' logic for {alone_mismatch} rows.")

    # Check Fare Per Person logic
    fare_calc = (df['ticket_price'] / df['family_size']).round(2)
    fare_diff = np.abs(df['fare_per_person'] - fare_calc).max()
    if fare_diff < 0.02:
        print("✓ PASS: 'fare_per_person' derived feature aligns with ticket_price and family_size.")
    else:
        print(f"⚠ WARNING: Discrepancy detected in 'fare_per_person' values (Max Diff: {fare_diff}).")

    # 5. Core Feature Relationships with Survival (Sanity Check)
    print("\n" + "-" * 60)
    print("4. FEATURE CORRELATIONS WITH SURVIVAL")
    print("-" * 60)
    
    class_survival = df.groupby('class')['survived'].mean()
    print("Survival Rate by Class:")
    for pclass, rate in class_survival.items():
        print(f"  Class {pclass}: {rate:.2%}")
        
    gender_survival = df.groupby('gender')['survived'].mean()
    print("\nSurvival Rate by Gender:")
    for g, rate in gender_survival.items():
        print(f"  {g}: {rate:.2%}")

    # Verify Class 1 > Class 3 survival
    if class_survival[1] > class_survival[3]:
        print("\n✓ PASS: Class 1 survival rate is strictly higher than Class 3.")
    else:
        print("\n⚠ WARNING: Expected Class 1 to have higher survival rate than Class 3.")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    validate_titanic_dataset(FILE_NAME)

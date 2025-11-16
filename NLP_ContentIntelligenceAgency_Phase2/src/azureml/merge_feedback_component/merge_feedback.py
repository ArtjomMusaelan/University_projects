import argparse
import os
import glob
import pandas as pd


def main():
    """
    Merge feedback CSV files with a training CSV file.

    Reads a training CSV file and all CSV files in a feedback directory,
    merges them (removing duplicates), and writes the result to an output CSV file.
    Accepts custom text and label column names. Handles feedback CSVs that might not
    have a label column.
    """
    parser = argparse.ArgumentParser(
        description="Merge feedback CSV files with a training CSV file."
    )
    parser.add_argument(
        "--train_csv", type=str, required=True,
        help="Path to the training CSV file."
    )
    parser.add_argument(
        "--feedback_dir", type=str, required=True,
        help="Directory containing feedback CSV files."
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Path to save the merged output CSV file."
    )
    parser.add_argument(
        "--text_col", type=str, required=True,
        help="Name of the text column in CSVs."
    )
    parser.add_argument(
        "--label_col", type=str, required=True,
        help="Name of the label column in CSVs."
    )
    args = parser.parse_args()

    # Read the training CSV file
    train_df = pd.read_csv(
        args.train_csv, dtype={args.text_col: str, args.label_col: str}
    )

    # Find all CSV files in the feedback directory
    feedback_files = glob.glob(os.path.join(args.feedback_dir, "*.csv"))

    dfs = []
    for f in feedback_files:
        df = pd.read_csv(f, dtype=str)
        # Only keep text_col and label_col if present
        cols = [col for col in [args.text_col, args.label_col] if col in df.columns]
        df = df[cols]
        # If label_col is missing, add it with empty string
        if args.label_col not in df.columns:
            df[args.label_col] = ""
        # Ensure text_col exists
        if args.text_col in df.columns:
            dfs.append(df[[args.text_col, args.label_col]])

    if dfs:
        all_feedback = pd.concat(dfs, ignore_index=True)
        merged = pd.concat([train_df, all_feedback], ignore_index=True)
        # Drop rows with missing or empty text_col
        merged = merged.dropna(subset=[args.text_col])
        merged = merged[merged[args.text_col].str.strip() != ""]
        # If label_col exists, drop rows with missing label_col
        if args.label_col in merged.columns:
            merged = merged[merged[args.label_col].notna()]
        # Remove duplicates based on available columns
        merged = merged.drop_duplicates(subset=[args.text_col, args.label_col])
    else:
        merged = train_df

    merged.to_csv(args.output_csv, index=False)
    print(f"Merged CSV saved: {args.output_csv} (rows: {len(merged)})")


if __name__ == "__main__":
    main()

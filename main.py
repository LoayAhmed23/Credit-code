"""
CLI entry point for the Credit Risk Prediction System.

Usage
-----
    python main.py train          # Train with default LightGBM params
    python main.py tune           # Train with hyperparameter tuning
    python main.py score          # Score new data with a saved model
"""

import argparse
import sys

from pipeline import run_training_pipeline, run_scoring_pipeline
import config


def main():
    parser = argparse.ArgumentParser(
        description="Credit Risk Prediction System",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    train_parser = sub.add_parser("train", help="Train the model with default hyperparameters")
    train_parser.add_argument("--sample", action="store_true", help="Use stratified 25% of data for rapid training/debugging")

    # --- tune ---
    tune_parser = sub.add_parser("tune", help="Train with RandomizedSearchCV hyperparameter tuning")
    tune_parser.add_argument("--sample", action="store_true", help="Use stratified 25% of data for rapid tuning/debugging")

    # --- score ---
    score_parser = sub.add_parser("score", help="Score new data using a saved model")
    score_parser.add_argument(
        "--model", default=config.MODEL_PATH,
        help="Path to saved model (.joblib)",
    )
    score_parser.add_argument(
        "--prime-dir", default=config.PRIME_DATA_DIR,
        help="Directory with new prime CSVs",
    )
    score_parser.add_argument(
        "--txn-dir", default=config.TRANSACTION_DATA_DIR,
        help="Directory with new transaction CSVs",
    )
    score_parser.add_argument(
        "--output", default=config.SCORES_PATH,
        help="Output CSV path for risk scores",
    )

    args = parser.parse_args()

    if args.command == "train":
        metrics = run_training_pipeline(tune=False, sample=args.sample)
        print("\nDone. Metrics:", metrics)

    elif args.command == "tune":
        metrics = run_training_pipeline(tune=True, sample=args.sample)
        print("\nDone. Metrics:", metrics)

    elif args.command == "score":
        run_scoring_pipeline(
            model_path=args.model,
            prime_dir=args.prime_dir,
            txn_dir=args.txn_dir,
            output_path=args.output,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

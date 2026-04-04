"""
Baby Activity Prediction - Baseline Model (LightGBM)

Trains 3 independent binary classifiers for sleep/nursing/diaper prediction.
Evaluates with AUC-ROC, PR-AUC, and prints feature importance.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from pathlib import Path
import pickle
import json

from data_preprocessing import get_feature_columns, get_target_columns

PROCESSED_DIR = Path(__file__).parent.parent / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models"


def load_data():
    train = pd.read_pickle(PROCESSED_DIR / "train.pkl")
    test = pd.read_pickle(PROCESSED_DIR / "test.pkl")
    return train, test


def train_model(X_train, y_train, target_name):
    """Train a LightGBM binary classifier."""
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / max(pos_count, 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos,
        "verbose": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    print(f"  [{target_name}] Trained with {model.num_trees()} trees, "
          f"pos/neg ratio: {pos_count}/{neg_count} ({scale_pos:.1f}x weight)")

    return model


def evaluate_model(model, X_test, y_test, target_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)

    metrics = {}
    if y_test.sum() > 0:
        metrics["auc_roc"] = roc_auc_score(y_test, y_pred)
        metrics["pr_auc"] = average_precision_score(y_test, y_pred)
    else:
        metrics["auc_roc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    # Binary predictions at 0.5 threshold
    y_pred_binary = (y_pred > 0.5).astype(int)
    metrics["accuracy"] = (y_pred_binary == y_test).mean()
    metrics["pos_rate_actual"] = y_test.mean()
    metrics["pos_rate_predicted"] = y_pred.mean()

    print(f"\n  [{target_name}] Results:")
    print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"    PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Actual positive rate:    {metrics['pos_rate_actual']:.4f}")
    print(f"    Predicted positive rate: {metrics['pos_rate_predicted']:.4f}")

    return metrics, y_pred


def get_feature_importance(model, feature_names):
    """Return feature importance as a sorted DataFrame."""
    importance = model.feature_importance(importance_type="gain")
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    return df


def main():
    print("=" * 60)
    print("Baby Activity Prediction - Baseline Model (LightGBM)")
    print("=" * 60)

    # 1. Load data
    train, test = load_data()
    feature_cols = get_feature_columns()
    target_cols = get_target_columns()

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Train & evaluate each target
    MODEL_DIR.mkdir(exist_ok=True)
    all_metrics = {}
    all_predictions = {}
    all_importances = {}

    for target in target_cols:
        event_name = target.replace("y_", "")
        print(f"\n{'─' * 40}")
        print(f"Training: {event_name}")

        y_train = train[target]
        y_test = test[target]

        model = train_model(X_train, y_train, event_name)
        metrics, y_pred = evaluate_model(model, X_test, y_test, event_name)
        importance = get_feature_importance(model, feature_cols)

        all_metrics[event_name] = metrics
        all_predictions[event_name] = y_pred
        all_importances[event_name] = importance

        # Save model
        model.save_model(str(MODEL_DIR / f"lgbm_{event_name}.txt"))

        # Print top features
        print(f"\n  Top 5 features ({event_name}):")
        for _, row in importance.head(5).iterrows():
            print(f"    {row['feature']:30s} {row['importance']:.1f}")

    # 3. Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"{'Event':>10s} | {'AUC-ROC':>8s} | {'PR-AUC':>8s} | {'Accuracy':>8s}")
    print("-" * 45)
    for event, m in all_metrics.items():
        print(f"{event:>10s} | {m['auc_roc']:>8.4f} | {m['pr_auc']:>8.4f} | {m['accuracy']:>8.4f}")

    # 4. Save predictions and metrics
    pred_df = test[["bin_time"]].copy()
    for event, preds in all_predictions.items():
        pred_df[f"prob_{event}"] = preds
        pred_df[f"actual_{event}"] = test[f"y_{event}"].values

    pred_df.to_pickle(PROCESSED_DIR / "predictions.pkl")

    with open(PROCESSED_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Save feature importances
    for event, imp in all_importances.items():
        imp.to_csv(PROCESSED_DIR / f"feature_importance_{event}.csv", index=False)

    print(f"\nSaved models to {MODEL_DIR}")
    print(f"Saved predictions and metrics to {PROCESSED_DIR}")

    return all_metrics


if __name__ == "__main__":
    main()

"""
Baby Activity Prediction - Evaluation & Visualization

Generates charts: feature importance, prediction probabilities over time,
ROC curves, and hourly event probability distribution.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import roc_curve, precision_recall_curve
from pathlib import Path
import json

PROCESSED_DIR = Path(__file__).parent.parent / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "figures"

EVENT_NAMES = ["sleep", "nursing", "diaper"]
EVENT_COLORS = {"sleep": "#4A90D9", "nursing": "#E67E22", "diaper": "#27AE60"}


def load_data():
    predictions = pd.read_pickle(PROCESSED_DIR / "predictions.pkl")
    with open(PROCESSED_DIR / "metrics.json") as f:
        metrics = json.load(f)
    importances = {}
    for event in EVENT_NAMES:
        importances[event] = pd.read_csv(PROCESSED_DIR / f"feature_importance_{event}.csv")
    return predictions, metrics, importances


def plot_feature_importance(importances):
    """Plot feature importance for each event type."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, event in zip(axes, EVENT_NAMES):
        imp = importances[event].head(10)
        ax.barh(range(len(imp)), imp["importance"].values, color=EVENT_COLORS[event], alpha=0.8)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp["feature"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{event.capitalize()} - Feature Importance", fontsize=12)
        ax.set_xlabel("Gain")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved feature_importance.png")


def plot_predictions_timeline(predictions):
    """Plot predicted probabilities over time for a sample period."""
    # Take a 7-day sample from the middle of test set
    pred = predictions.copy()
    pred["bin_time"] = pd.to_datetime(pred["bin_time"])

    mid = pred["bin_time"].iloc[len(pred) // 2]
    start = mid - pd.Timedelta(days=3)
    end = mid + pd.Timedelta(days=4)
    sample = pred[(pred["bin_time"] >= start) & (pred["bin_time"] <= end)]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for ax, event in zip(axes, EVENT_NAMES):
        prob_col = f"prob_{event}"
        actual_col = f"actual_{event}"

        ax.fill_between(sample["bin_time"], sample[prob_col],
                        alpha=0.3, color=EVENT_COLORS[event], label="Predicted probability")
        ax.plot(sample["bin_time"], sample[prob_col],
                color=EVENT_COLORS[event], linewidth=0.8, alpha=0.7)

        # Mark actual events
        actual_events = sample[sample[actual_col] == 1]
        if len(actual_events) > 0:
            ax.scatter(actual_events["bin_time"],
                       [1.0] * len(actual_events),
                       color="red", marker="v", s=50, zorder=5, label="Actual event")

        ax.set_ylabel(f"P({event})", fontsize=11)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{event.capitalize()} Prediction", fontsize=12)

        # Add day/night shading
        for day_offset in range(-4, 5):
            night_start = start.normalize() + pd.Timedelta(days=day_offset, hours=21)
            night_end = start.normalize() + pd.Timedelta(days=day_offset + 1, hours=6)
            ax.axvspan(night_start, night_end, alpha=0.08, color="navy")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    plt.suptitle(f"Prediction Timeline ({start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved predictions_timeline.png")


def plot_roc_pr_curves(predictions):
    """Plot ROC and PR curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curves
    for event in EVENT_NAMES:
        y_true = predictions[f"actual_{event}"]
        y_pred = predictions[f"prob_{event}"]
        if y_true.sum() > 0:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            axes[0].plot(fpr, tpr, color=EVENT_COLORS[event], linewidth=2, label=event.capitalize())

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend()

    # PR curves
    for event in EVENT_NAMES:
        y_true = predictions[f"actual_{event}"]
        y_pred = predictions[f"prob_{event}"]
        if y_true.sum() > 0:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            axes[1].plot(recall, precision, color=EVENT_COLORS[event], linewidth=2, label=event.capitalize())

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved roc_pr_curves.png")


def plot_hourly_distribution(predictions):
    """Plot average predicted probability by hour of day."""
    pred = predictions.copy()
    pred["hour"] = pd.to_datetime(pred["bin_time"]).dt.hour

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(24)
    width = 0.25

    for idx, event in enumerate(EVENT_NAMES):
        hourly_mean = pred.groupby("hour")[f"prob_{event}"].mean()
        ax.bar(x + idx * width, hourly_mean.values, width,
               color=EVENT_COLORS[event], alpha=0.8, label=event.capitalize())

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Predicted Probability")
    ax.set_title("Predicted Event Probability by Hour of Day")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hourly_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved hourly_distribution.png")


def main():
    print("=" * 60)
    print("Baby Activity Prediction - Evaluation & Visualization")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    predictions, metrics, importances = load_data()

    print(f"\nTest predictions: {len(predictions)} time bins")

    print("\nGenerating visualizations...")
    plot_feature_importance(importances)
    plot_predictions_timeline(predictions)
    plot_roc_pr_curves(predictions)
    plot_hourly_distribution(predictions)

    print(f"\nAll figures saved to {OUTPUT_DIR}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Final Metrics Summary")
    print(f"{'=' * 60}")
    for event, m in metrics.items():
        print(f"  {event:>10s}: AUC-ROC={m['auc_roc']:.4f}, PR-AUC={m['pr_auc']:.4f}")


if __name__ == "__main__":
    main()

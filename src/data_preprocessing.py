"""
Baby Activity Prediction - Data Preprocessing Pipeline

Loads sleep/nursing/diaper CSVs for baby_1, merges into unified timeline,
creates time-binned features, and outputs train/test datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "processed"
BABY_ID = "baby_1"
BIN_MINUTES = 30
TRAIN_RATIO = 0.8
# baby_1 nursing stops ~2021.03, diaper ~2022.05. Use period with all 3 active.
CUTOFF_DATE = "2021-02-28"


def load_baby1_events():
    """Load and merge all event types for baby_1 into a unified timeline."""

    # --- Sleep ---
    df_sleep = pd.read_csv(DATA_DIR / "baby_data_sleep.csv")
    df_sleep = df_sleep[df_sleep["baby_id"] == BABY_ID].copy()
    df_sleep["datetime"] = pd.to_datetime(df_sleep["combined_time"], format="mixed")
    df_sleep["event_type"] = "sleep"
    df_sleep = df_sleep.rename(columns={"total_sleep_min": "duration_min"})
    df_sleep["detail"] = None
    events_sleep = df_sleep[["datetime", "event_type", "duration_min", "detail"]]

    # --- Nursing ---
    df_nursing = pd.read_csv(DATA_DIR / "baby_data_nursing.csv")
    df_nursing = df_nursing[df_nursing["baby_id"] == BABY_ID].copy()
    df_nursing["datetime"] = pd.to_datetime(df_nursing["combined_time"], format="mixed")
    df_nursing["event_type"] = "nursing"
    df_nursing = df_nursing.rename(columns={
        "total_feeding_duration_min": "duration_min",
        "feeding-side": "detail",
    })
    events_nursing = df_nursing[["datetime", "event_type", "duration_min", "detail"]]

    # --- Diaper ---
    df_diaper = pd.read_csv(DATA_DIR / "baby_data_diaper.csv")
    df_diaper = df_diaper[df_diaper["baby_id"] == BABY_ID].copy()
    df_diaper["datetime"] = pd.to_datetime(df_diaper["combined_time"], format="mixed")
    df_diaper["event_type"] = "diaper"
    df_diaper["duration_min"] = 0
    df_diaper = df_diaper.rename(columns={"diaper_type": "detail"})
    events_diaper = df_diaper[["datetime", "event_type", "duration_min", "detail"]]

    # Merge & sort
    events = pd.concat([events_sleep, events_nursing, events_diaper], ignore_index=True)
    events = events.sort_values("datetime").reset_index(drop=True)

    print(f"Loaded {len(events)} events for {BABY_ID}")
    print(f"  Sleep: {len(events_sleep)}, Nursing: {len(events_nursing)}, Diaper: {len(events_diaper)}")
    print(f"  Date range: {events.datetime.min()} ~ {events.datetime.max()}")

    return events


def create_time_bins(events):
    """Create time bins covering the full event range."""
    start = events["datetime"].min().floor("D")
    end = events["datetime"].max().ceil("D")
    bins = pd.date_range(start=start, end=end, freq=f"{BIN_MINUTES}min")
    return bins


def assign_events_to_bins(events, bins):
    """Assign each event to its nearest time bin."""
    bin_index = np.searchsorted(bins, events["datetime"].values) - 1
    bin_index = np.clip(bin_index, 0, len(bins) - 2)
    events = events.copy()
    events["bin_idx"] = bin_index
    events["bin_time"] = bins[bin_index]
    return events


def build_features(events, bins):
    """Build feature matrix for each time bin."""
    events_binned = assign_events_to_bins(events, bins)

    # Pre-compute event arrays for fast lookup
    sleep_events = events[events["event_type"] == "sleep"].copy()
    nursing_events = events[events["event_type"] == "nursing"].copy()
    diaper_events = events[events["event_type"] == "diaper"].copy()

    first_event_time = events["datetime"].min()
    n_bins = len(bins) - 1

    # Target labels: which events occur in each bin
    sleep_bins = set(events_binned[events_binned["event_type"] == "sleep"]["bin_idx"].values)
    nursing_bins = set(events_binned[events_binned["event_type"] == "nursing"]["bin_idx"].values)
    diaper_bins = set(events_binned[events_binned["event_type"] == "diaper"]["bin_idx"].values)

    # Build features for each bin
    records = []

    # Convert to numpy for speed
    sleep_times = sleep_events["datetime"].values.astype("datetime64[m]").astype(np.float64)
    nursing_times = nursing_events["datetime"].values.astype("datetime64[m]").astype(np.float64)
    diaper_times = diaper_events["datetime"].values.astype("datetime64[m]").astype(np.float64)

    sleep_durations = sleep_events["duration_min"].values.astype(np.float64)
    nursing_durations = nursing_events["duration_min"].values.astype(np.float64)

    diaper_details = diaper_events["detail"].values

    # Diaper type encoding
    diaper_type_map = {"Wet": 0, "Dirty": 1, "Mixed": 2, "Dry": 3}

    for i in range(n_bins):
        bin_time = bins[i]
        bin_ts = pd.Timestamp(bin_time)
        bin_minutes = np.float64(bin_ts.value / 1e9 / 60)

        # --- Temporal features ---
        hour = bin_ts.hour + bin_ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        dow = bin_ts.dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        age_days = (bin_ts - first_event_time).total_seconds() / 86400

        # --- Recency features ---
        # Time since last event of each type
        past_sleep = sleep_times[sleep_times <= bin_minutes]
        past_nursing = nursing_times[nursing_times <= bin_minutes]
        past_diaper = diaper_times[diaper_times <= bin_minutes]

        time_since_last_sleep = (bin_minutes - past_sleep[-1]) if len(past_sleep) > 0 else -1
        time_since_last_nursing = (bin_minutes - past_nursing[-1]) if len(past_nursing) > 0 else -1
        time_since_last_diaper = (bin_minutes - past_diaper[-1]) if len(past_diaper) > 0 else -1

        # Last event durations
        past_sleep_idx = np.searchsorted(sleep_times, bin_minutes, side="right") - 1
        last_sleep_dur = sleep_durations[past_sleep_idx] if past_sleep_idx >= 0 else 0

        past_nursing_idx = np.searchsorted(nursing_times, bin_minutes, side="right") - 1
        last_nursing_dur = nursing_durations[past_nursing_idx] if past_nursing_idx >= 0 else 0

        past_diaper_idx = np.searchsorted(diaper_times, bin_minutes, side="right") - 1
        last_diaper_type_val = diaper_type_map.get(
            str(diaper_details[past_diaper_idx]) if past_diaper_idx >= 0 else "", -1
        )

        # --- Rolling window features (24h) ---
        window_24h = 24 * 60  # 24 hours in minutes
        sleep_count_24h = np.sum((bin_minutes - past_sleep) <= window_24h) if len(past_sleep) > 0 else 0
        nursing_count_24h = np.sum((bin_minutes - past_nursing) <= window_24h) if len(past_nursing) > 0 else 0
        diaper_count_24h = np.sum((bin_minutes - past_diaper) <= window_24h) if len(past_diaper) > 0 else 0

        # --- Targets ---
        y_sleep = 1 if i in sleep_bins else 0
        y_nursing = 1 if i in nursing_bins else 0
        y_diaper = 1 if i in diaper_bins else 0

        records.append({
            "bin_time": bin_ts,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "age_days": age_days,
            "time_since_last_sleep": time_since_last_sleep,
            "time_since_last_nursing": time_since_last_nursing,
            "time_since_last_diaper": time_since_last_diaper,
            "last_sleep_duration": last_sleep_dur,
            "last_nursing_duration": last_nursing_dur,
            "last_diaper_type": last_diaper_type_val,
            "sleep_count_24h": sleep_count_24h,
            "nursing_count_24h": nursing_count_24h,
            "diaper_count_24h": diaper_count_24h,
            "y_sleep": y_sleep,
            "y_nursing": y_nursing,
            "y_diaper": y_diaper,
        })

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{n_bins} bins...")

    df = pd.DataFrame(records)
    return df


def filter_active_period(df):
    """Filter out bins where no events have occurred yet (before first event)."""
    return df[df["time_since_last_sleep"] >= 0].copy()


def time_based_split(df, train_ratio=TRAIN_RATIO):
    """Split by time: first train_ratio as train, rest as test."""
    n = len(df)
    split_idx = int(n * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def get_feature_columns():
    """Return list of feature column names."""
    return [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "age_days",
        "time_since_last_sleep", "time_since_last_nursing", "time_since_last_diaper",
        "last_sleep_duration", "last_nursing_duration",
        "last_diaper_type",
        "sleep_count_24h", "nursing_count_24h", "diaper_count_24h",
    ]


def get_target_columns():
    return ["y_sleep", "y_nursing", "y_diaper"]


def main():
    print("=" * 60)
    print("Baby Activity Prediction - Data Preprocessing")
    print("=" * 60)

    # 1. Load events
    events = load_baby1_events()

    # 1b. Trim to active period (all 3 event types present)
    events = events[events["datetime"] <= CUTOFF_DATE].copy()
    print(f"\nAfter trimming to <= {CUTOFF_DATE}: {len(events)} events")
    for evt in ["sleep", "nursing", "diaper"]:
        print(f"  {evt}: {(events.event_type == evt).sum()}")

    # 2. Create time bins
    bins = create_time_bins(events)
    print(f"\nCreated {len(bins)-1} time bins ({BIN_MINUTES}min each)")
    print(f"  From {bins[0]} to {bins[-1]}")

    # 3. Build features
    print("\nBuilding features...")
    df = build_features(events, bins)
    print(f"  Raw feature matrix: {df.shape}")

    # 4. Filter active period
    df = filter_active_period(df)
    print(f"  After filtering inactive period: {df.shape}")

    # 5. Split
    train, test = time_based_split(df)
    print(f"\nTrain/Test split:")
    print(f"  Train: {train.shape} ({train.bin_time.min()} ~ {train.bin_time.max()})")
    print(f"  Test:  {test.shape} ({test.bin_time.min()} ~ {test.bin_time.max()})")

    # 6. Class distribution
    for target in get_target_columns():
        train_pos = train[target].sum()
        test_pos = test[target].sum()
        print(f"  {target}: train={train_pos}/{len(train)} ({train_pos/len(train)*100:.2f}%), "
              f"test={test_pos}/{len(test)} ({test_pos/len(test)*100:.2f}%)")

    # 7. Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    train.to_pickle(OUTPUT_DIR / "train.pkl")
    test.to_pickle(OUTPUT_DIR / "test.pkl")
    print(f"\nSaved to {OUTPUT_DIR}")

    return train, test


if __name__ == "__main__":
    main()

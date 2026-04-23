from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pygrib
from scipy.ndimage import maximum_filter


DEFAULT_ARCHIVE_ROOT = Path("/var/data/mrms_grib_archive")
DEFAULT_OUTPUT_PATH = Path("/var/data/ai_models/radar_severe_model.json")
DEFAULT_SAMPLES_PER_CLASS = 64
DEFAULT_RANDOM_SEED = 42
DEFAULT_FORECAST_HOURS = 2.0
DEFAULT_FUTURE_NEIGHBORHOOD_RADIUS = 8
MAX_DBZ = 80.0
POSITIVE_DBZ_THRESHOLD = 45.0
NEGATIVE_DBZ_THRESHOLD = 20.0
INVALID_DBZ_THRESHOLD = -99.0
FEATURE_NAMES = [
    "center_dbz",
    "window_mean_dbz",
    "window_max_dbz",
    "window_std_dbz",
    "fraction_ge_35dbz",
    "fraction_ge_50dbz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight severe-likelihood model from archived MRMS GRIB files."
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help="Directory containing archived .grib2 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output file for the trained model artifact.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="Maximum positive and negative samples to draw from each GRIB file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--forecast-hours",
        type=float,
        default=DEFAULT_FORECAST_HOURS,
        help="Forecast lead window used to label future severe outcomes. Default: 2.0.",
    )
    parser.add_argument(
        "--future-neighborhood-radius",
        type=int,
        default=DEFAULT_FUTURE_NEIGHBORHOOD_RADIUS,
        help="Neighborhood radius in pixels used when looking for future severe reflectivity. Default: 8.",
    )
    return parser.parse_args()


def list_grib_files(archive_root: Path) -> list[Path]:
    if not archive_root.exists():
        return []
    return sorted(archive_root.rglob("*.grib2"))


def load_reflectivity(path: Path) -> tuple[np.ndarray, str]:
    grib_file = pygrib.open(str(path))
    try:
        message = grib_file.message(1)
        values = np.asarray(message.values, dtype=np.float32)
        valid_time = message.validDate.replace(tzinfo=timezone.utc).isoformat()
    finally:
        grib_file.close()
    return values, valid_time


def parse_valid_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def sample_positions(mask: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    positions = np.argwhere(mask)
    if positions.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if len(positions) <= max_count:
        return positions.astype(np.int32, copy=False)
    chosen = rng.choice(len(positions), size=max_count, replace=False)
    return positions[chosen].astype(np.int32, copy=False)


def extract_features(values: np.ndarray, row: int, col: int, radius: int = 4) -> np.ndarray | None:
    row_start = max(0, row - radius)
    row_end = min(values.shape[0], row + radius + 1)
    col_start = max(0, col - radius)
    col_end = min(values.shape[1], col + radius + 1)

    window = values[row_start:row_end, col_start:col_end]
    valid_window = window[window > INVALID_DBZ_THRESHOLD]
    if valid_window.size == 0:
        return None

    valid_window = np.clip(valid_window, 0.0, MAX_DBZ)
    center_dbz = float(np.clip(values[row, col], 0.0, MAX_DBZ))
    return np.array(
        [
            center_dbz,
            float(valid_window.mean()),
            float(valid_window.max()),
            float(valid_window.std()),
            float(np.mean(valid_window >= 35.0)),
            float(np.mean(valid_window >= 50.0)),
        ],
        dtype=np.float32,
    )


def is_positive_example(features: np.ndarray) -> bool:
    center_dbz, window_mean, window_max, _, fraction_ge_35, fraction_ge_50 = features
    return bool(
        window_max >= 52.0
        and (center_dbz >= 46.0 or window_mean >= 36.0)
        and (fraction_ge_35 >= 0.18 or fraction_ge_50 >= 0.05)
    )


def is_negative_example(features: np.ndarray) -> bool:
    _, window_mean, window_max, _, fraction_ge_35, fraction_ge_50 = features
    return bool(window_max < 30.0 and window_mean < 20.0 and fraction_ge_35 == 0.0 and fraction_ge_50 == 0.0)


def build_future_target_grid(
    future_values: np.ndarray,
    neighborhood_radius: int,
) -> np.ndarray:
    valid_future = np.where(future_values > INVALID_DBZ_THRESHOLD, np.clip(future_values, 0.0, MAX_DBZ), 0.0)
    size = (neighborhood_radius * 2) + 1
    return maximum_filter(valid_future, size=size, mode="constant", cval=0.0).astype(np.float32)


def future_positive_label(future_max_dbz: float) -> bool:
    return future_max_dbz >= 50.0


def future_negative_label(future_max_dbz: float) -> bool:
    return future_max_dbz < 30.0


def collect_training_rows(
    current_values: np.ndarray,
    future_target_grid: np.ndarray,
    samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    positive_candidates = sample_positions(
        (current_values >= POSITIVE_DBZ_THRESHOLD) | (future_target_grid >= 50.0),
        samples_per_class * 6,
        rng,
    )
    negative_candidates = sample_positions(
        (current_values >= 0.0) & (current_values < NEGATIVE_DBZ_THRESHOLD) & (future_target_grid < 30.0),
        samples_per_class * 6,
        rng,
    )

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    positive_count = 0
    for row, col in positive_candidates:
        if positive_count >= samples_per_class:
            break
        features = extract_features(current_values, int(row), int(col))
        if features is None:
            continue
        future_max_dbz = float(future_target_grid[int(row), int(col)])
        if not future_positive_label(future_max_dbz):
            continue
        feature_rows.append(features)
        labels.append(1)
        positive_count += 1

    negative_count = 0
    for row, col in negative_candidates:
        if negative_count >= samples_per_class:
            break
        features = extract_features(current_values, int(row), int(col))
        if features is None:
            continue
        future_max_dbz = float(future_target_grid[int(row), int(col)])
        if not future_negative_label(future_max_dbz):
            continue
        feature_rows.append(features)
        labels.append(0)
        negative_count += 1

    if not feature_rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.vstack(feature_rows).astype(np.float32), np.asarray(labels, dtype=np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def train_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 350,
    learning_rate: float = 0.12,
    l2_penalty: float = 0.0005,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    normalized_features = (features - feature_mean) / feature_scale

    weights = np.zeros(normalized_features.shape[1], dtype=np.float64)
    bias = 0.0
    sample_count = float(len(labels))

    for _ in range(epochs):
        logits = normalized_features @ weights + bias
        predictions = sigmoid(logits)
        error = predictions - labels
        gradient_weights = (normalized_features.T @ error) / sample_count + (l2_penalty * weights)
        gradient_bias = float(error.mean())

        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    return weights.astype(np.float32), float(bias), feature_mean.astype(np.float32), feature_scale.astype(np.float32)


def evaluate_model(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[dict[str, float], float]:
    normalized_features = (features - feature_mean) / feature_scale
    probabilities = sigmoid(normalized_features @ weights + bias)

    best_metrics: dict[str, float] = {}
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.35, 0.71, 0.05):
        predicted = probabilities >= threshold
        true_positive = int(np.sum((predicted == 1) & (labels == 1)))
        false_positive = int(np.sum((predicted == 1) & (labels == 0)))
        false_negative = int(np.sum((predicted == 0) & (labels == 1)))
        true_negative = int(np.sum((predicted == 0) & (labels == 0)))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        accuracy = (true_positive + true_negative) / max(len(labels), 1)
        if precision + recall:
            f1_score = 2.0 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = float(round(float(threshold), 2))
            best_metrics = {
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1_score), 4),
            }

    return best_metrics, best_threshold


def build_dataset(
    grib_files: list[Path],
    samples_per_class: int,
    forecast_hours: float,
    future_neighborhood_radius: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    valid_times: list[str] = []
    files_with_samples = 0

    loaded_frames = [(path, *load_reflectivity(path)) for path in grib_files]

    for index, (grib_path, values, valid_time) in enumerate(loaded_frames, start=1):
        current_time = parse_valid_time(valid_time)
        future_frames = [
            future_values
            for _, future_values, future_time in loaded_frames[index:]
            if 0.0 < (parse_valid_time(future_time) - current_time).total_seconds() / 3600.0 <= forecast_hours
        ]
        if not future_frames:
            if index == 1 or index % 100 == 0 or index == len(loaded_frames):
                print(f"Processed {index} / {len(loaded_frames)} GRIB files")
            continue

        future_stack = np.stack(future_frames)
        future_max_grid = build_future_target_grid(np.max(future_stack, axis=0), neighborhood_radius=future_neighborhood_radius)
        features, labels = collect_training_rows(
            values,
            future_max_grid,
            samples_per_class=samples_per_class,
            rng=rng,
        )
        if len(labels):
            feature_batches.append(features)
            label_batches.append(labels)
            valid_times.append(valid_time)
            files_with_samples += 1

        if index == 1 or index % 100 == 0 or index == len(loaded_frames):
            print(f"Processed {index} / {len(loaded_frames)} GRIB files")

    if not feature_batches:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            valid_times,
            files_with_samples,
        )

    return np.vstack(feature_batches), np.concatenate(label_batches), valid_times, files_with_samples


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    if len(indices) < 10:
        return features, features, labels, labels

    split_index = int(len(indices) * 0.8)
    split_index = min(max(split_index, 1), len(indices) - 1)
    train_indices = indices[:split_index]
    validation_indices = indices[split_index:]
    return features[train_indices], features[validation_indices], labels[train_indices], labels[validation_indices]


def save_model(
    output_path: Path,
    archive_root: Path,
    weights: np.ndarray,
    bias: float,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    metrics: dict[str, float],
    decision_threshold: float,
    grib_files: list[Path],
    files_with_samples: int,
    labels: np.ndarray,
    valid_times: list[str],
    samples_per_class: int,
    forecast_hours: float,
    future_neighborhood_radius: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))

    payload = {
        "artifactVersion": 1,
        "modelName": "MRMS Future Severe Outlook Learner",
        "modelType": "logistic_regression",
        "targetType": "future_severe_outlook",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "archiveRoot": str(archive_root),
        "featureNames": FEATURE_NAMES,
        "featureMean": [round(float(value), 6) for value in feature_mean],
        "featureScale": [round(float(value), 6) for value in feature_scale],
        "weights": [round(float(value), 6) for value in weights],
        "bias": round(float(bias), 6),
        "decisionThreshold": decision_threshold,
        "trainingSummary": {
            "gribFilesSeen": len(grib_files),
            "gribFilesUsed": files_with_samples,
            "sampleCount": int(len(labels)),
            "positiveSamples": positive_count,
            "negativeSamples": negative_count,
            "samplesPerClassPerFile": samples_per_class,
            "forecastLeadHours": forecast_hours,
            "futureNeighborhoodRadiusPixels": future_neighborhood_radius,
            "firstValidTime": valid_times[0] if valid_times else None,
            "lastValidTime": valid_times[-1] if valid_times else None,
            "validationMetrics": metrics,
        },
        "notes": [
            "This model is trained on future radar outcomes rather than current reflectivity intensity.",
            "Positive labels mean the location is followed by strong nearby reflectivity within the forecast lead window.",
            "Better results still require more archived GRIB history and real labels such as warnings, reports, or analyst-drawn outlines.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise SystemExit("--samples-per-class must be greater than 0")
    if args.forecast_hours <= 0:
        raise SystemExit("--forecast-hours must be greater than 0")
    if args.future_neighborhood_radius <= 0:
        raise SystemExit("--future-neighborhood-radius must be greater than 0")

    grib_files = list_grib_files(args.archive_root)
    if len(grib_files) < 2:
        raise SystemExit(f"Need at least 2 GRIB files to train a model. Found {len(grib_files)}.")

    rng = np.random.default_rng(args.seed)
    features, labels, valid_times, files_with_samples = build_dataset(
        grib_files,
        samples_per_class=args.samples_per_class,
        forecast_hours=args.forecast_hours,
        future_neighborhood_radius=args.future_neighborhood_radius,
        rng=rng,
    )
    if len(labels) < 20:
        raise SystemExit("Training archive did not yield enough usable samples.")

    train_features, validation_features, train_labels, validation_labels = split_dataset(features, labels, rng)
    weights, bias, feature_mean, feature_scale = train_logistic_regression(train_features, train_labels)
    metrics, decision_threshold = evaluate_model(
        validation_features,
        validation_labels,
        weights,
        bias,
        feature_mean,
        feature_scale,
    )

    save_model(
        output_path=args.output,
        archive_root=args.archive_root,
        weights=weights,
        bias=bias,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        metrics=metrics,
        decision_threshold=decision_threshold,
        grib_files=grib_files,
        files_with_samples=files_with_samples,
        labels=labels,
        valid_times=valid_times,
        samples_per_class=args.samples_per_class,
        forecast_hours=args.forecast_hours,
        future_neighborhood_radius=args.future_neighborhood_radius,
    )

    print(f"Saved trained model artifact to {args.output}")
    print(f"Samples: {len(labels)} | Files used: {files_with_samples} / {len(grib_files)} | Validation F1: {metrics.get('f1', 0.0)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_geometry.compare_candidate_models import (
    fit_logistic_features,
    fit_mlp,
    fold_metrics,
    linear,
    sample_weights,
    threshold_from_real,
)


def normalized_patches(patches: np.ndarray) -> np.ndarray:
    patches = patches.copy()
    patches[:, :, :, (0, 15)] = 0
    border = np.ones((16, 16), dtype=bool)
    border[4:12, 4:12] = False
    context = patches[:, :, border]
    center = np.median(context, axis=2)
    scale = np.median(np.abs(context - center[:, :, None]), axis=2) * 1.4826
    scale = np.maximum(scale, 0.05)
    return np.clip(
        (patches - center[:, :, None, None]) / scale[:, :, None, None],
        -6,
        6,
    )


def invariant_descriptor(patches: np.ndarray) -> np.ndarray:
    normalized = normalized_patches(patches)

    features = []
    for start, stop in ((2, 14), (4, 12), (6, 10)):
        crop = normalized[:, :, start:stop, start:stop]
        features.extend((
            crop.mean(axis=(2, 3)),
            crop.std(axis=(2, 3)),
            crop.max(axis=(2, 3)),
            np.quantile(crop, 0.75, axis=(2, 3)),
        ))

    pooled = normalized.reshape(len(patches), 3, 4, 4, 4, 4).mean(axis=(3, 5))
    features.append(pooled.reshape(len(patches), -1))
    row_mean = normalized.mean(axis=3)
    column_mean = normalized.mean(axis=2)
    features.extend((row_mean.reshape(len(patches), -1), column_mean.reshape(len(patches), -1)))
    features.extend((
        np.abs(normalized - normalized[:, :, :, ::-1]).mean(axis=(2, 3)),
        np.abs(normalized - normalized[:, :, ::-1, :]).mean(axis=(2, 3)),
    ))
    horizontal_gradient = np.diff(normalized, axis=3)
    vertical_gradient = np.diff(normalized, axis=2)
    features.extend((
        np.abs(horizontal_gradient).mean(axis=(2, 3)),
        np.abs(horizontal_gradient).max(axis=(2, 3)),
        np.abs(vertical_gradient).mean(axis=(2, 3)),
        np.abs(vertical_gradient).max(axis=(2, 3)),
    ))
    return np.column_stack(features).astype(np.float64)


def random_convolution_descriptor(patches: np.ndarray) -> np.ndarray:
    normalized = normalized_patches(patches).astype(np.float32)
    rng = np.random.default_rng(20260928)
    features = []
    for dilation in (1, 2, 3):
        kernels = rng.normal(0, 1, (16, 3, 3, 3)).astype(np.float32)
        kernels -= kernels.mean(axis=(1, 2, 3), keepdims=True)
        kernels /= np.maximum(
            np.sqrt((kernels * kernels).sum(axis=(1, 2, 3), keepdims=True)),
            1e-6,
        )
        span = 2 * dilation + 1
        windows = np.lib.stride_tricks.sliding_window_view(
            normalized, (span, span), axis=(-2, -1)
        )[..., ::dilation, ::dilation]
        response = np.einsum("nchwij,fcij->nfhw", windows, kernels, optimize=True)
        center = response[:, :, response.shape[2] // 4:3 * response.shape[2] // 4,
                          response.shape[3] // 4:3 * response.shape[3] // 4]
        features.extend((
            response.mean(axis=(2, 3)),
            response.std(axis=(2, 3)),
            response.max(axis=(2, 3)),
            (response > 0).mean(axis=(2, 3)),
            center.mean(axis=(2, 3)),
            center.max(axis=(2, 3)),
        ))
    return np.column_stack(features).astype(np.float64)


def randomized_patches(
    patches: np.ndarray,
    seed: int,
    mild: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    augmented = patches.copy()
    density_low = 0.78 if mild else 0.55
    geometry_low, geometry_high = ((0.94, 1.06) if mild else (0.85, 1.15))
    augmented[:, 0] *= rng.uniform(density_low, 1.15, (len(patches), 1, 1))
    augmented[:, 1:] *= rng.uniform(
        geometry_low, geometry_high, (len(patches), 1, 1, 1)
    )
    augmented[:, 1:] += rng.normal(
        0, 0.006 if mild else 0.015, augmented[:, 1:].shape
    )
    dropout = rng.random((len(patches), 1, 16, 16)) < (0.03 if mild else 0.08)
    augmented = np.where(dropout, 0, augmented)
    for index in range(len(augmented)):
        shift = rng.integers(-1, 2, size=2)
        if mild and rng.random() < 0.65:
            shift[:] = 0
        row_shift, column_shift = shift
        augmented[index] = np.roll(
            augmented[index], (row_shift, column_shift), axis=(1, 2)
        )
        if row_shift > 0:
            augmented[index, :, :row_shift] = 0
        elif row_shift < 0:
            augmented[index, :, row_shift:] = 0
        if column_shift > 0:
            augmented[index, :, :, :column_shift] = 0
        elif column_shift < 0:
            augmented[index, :, :, column_shift:] = 0
        if not mild and rng.random() < 0.20:
            augmented[index, :, rng.integers(0, 16)] = 0
    return np.clip(augmented, 0, 1).astype(np.float32)


def transplant_real_object(
    donors: np.ndarray,
    backgrounds: np.ndarray,
    seed: int,
    copies: int = 4,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = []
    for donor in donors:
        central = np.zeros((16, 16), dtype=bool)
        central[4:12, 4:12] = True
        signal = central & (donor[2] >= 0.05)
        padded = np.pad(signal, 1)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        signal = windows.any(axis=(2, 3)) & central
        for _ in range(copies):
            background = backgrounds[rng.integers(0, len(backgrounds))].copy()
            shift = rng.integers(-2, 3, size=2)
            shifted_signal = np.roll(signal, shift, axis=(0, 1))
            shifted_donor = np.roll(donor, shift, axis=(1, 2))
            keep = rng.random((16, 16)) >= rng.uniform(0.05, 0.25)
            shifted_signal &= keep
            background[:, shifted_signal] = np.maximum(
                background[:, shifted_signal], shifted_donor[:, shifted_signal]
            )
            output.append(background)
    return np.asarray(output, dtype=np.float32)


def summarize(folds: list[dict], feature_count: int) -> dict:
    return {
        "features": feature_count,
        "folds": folds,
        "mean_positive_recall": float(np.mean([row["positive_recall"] for row in folds])),
        "worst_positive_recall": float(np.min([row["positive_recall"] for row in folds])),
        "mean_synthetic_negative_specificity": float(np.mean([
            row["synthetic_negative_specificity"] for row in folds
        ])),
        "mean_real_frame_alarm_rate": float(np.mean([
            row["real_frame_alarm_rate"] for row in folds
        ])),
        "worst_real_frame_alarm_rate": float(np.max([
            row["real_frame_alarm_rate"] for row in folds
        ])),
    }


def evaluate(
    cache: Path,
    output: Path,
    osdar_cache: Path | None = None,
    weak_cache: Path | None = None,
) -> None:
    data = np.load(cache, allow_pickle=False)
    component = data["features"].astype(np.float64)
    invariant = invariant_descriptor(data["patches"])
    random_conv = random_convolution_descriptor(data["patches"])
    randomized_conv = [
        random_convolution_descriptor(randomized_patches(data["patches"], seed))
        for seed in (20260929, 20260930)
    ]
    mild_randomized_conv = [
        random_convolution_descriptor(
            randomized_patches(data["patches"], seed, mild=True)
        )
        for seed in (20261001, 20261002)
    ]
    labels = data["labels"]
    groups = data["groups"]
    records = json.loads(str(data["records_json"]))
    models = {
        "component_linear": (
            component,
            lambda x, y, w: fit_logistic_features(x, y, w, linear, steps=1500, l2=0.06),
        ),
        "invariant_cv": (
            invariant,
            lambda x, y, w: fit_logistic_features(x, y, w, linear, steps=1800, l2=0.12),
        ),
        "component_plus_invariant": (
            np.column_stack((component, invariant)),
            lambda x, y, w: fit_logistic_features(x, y, w, linear, steps=1800, l2=0.12),
        ),
        "invariant_mlp": (
            invariant,
            lambda x, y, w: fit_mlp(x, y, w, hidden=12, steps=1600, l2=0.05),
        ),
        "random_conv": (
            random_conv,
            lambda x, y, w: fit_logistic_features(x, y, w, linear, steps=1800, l2=0.10),
        ),
        "component_plus_random_conv": (
            np.column_stack((component, random_conv)),
            lambda x, y, w: fit_logistic_features(x, y, w, linear, steps=1800, l2=0.10),
        ),
    }
    report = {
        "protocol": {
            "split": "leave-one-recording-out",
            "normalization": "per-candidate outer-ring median and MAD",
            "features": "multi-scale center contrast, pooled layout, projections, symmetry, gradients",
            "intensity_used": False,
            "threshold": "99th percentile of training real-frame maximum scores",
        },
        "samples": len(labels),
        "models": {},
    }
    for name, (matrix, fitter) in models.items():
        folds = []
        for held_out in sorted(set(groups.tolist())):
            train = groups != held_out
            test = ~train
            train_records = [record for record, keep in zip(records, train) if keep]
            test_records = [record for record, keep in zip(records, test) if keep]
            predictor = fitter(matrix[train], labels[train], sample_weights(labels[train], train_records))
            threshold = threshold_from_real(predictor.score(matrix[train]), train_records)
            folds.append({
                "held_out_bag": held_out,
                **fold_metrics(predictor.score(matrix[test]), labels[test], test_records, threshold),
            })
        report["models"][name] = summarize(folds, matrix.shape[1])
    augmented_folds = []
    for held_out in sorted(set(groups.tolist())):
        train = groups != held_out
        test = ~train
        train_records = [record for record, keep in zip(records, train) if keep]
        augmented_records = train_records * 3
        augmented_labels = np.tile(labels[train], 3)
        augmented_matrix = np.vstack((
            random_conv[train],
            randomized_conv[0][train],
            randomized_conv[1][train],
        ))
        predictor = fit_logistic_features(
            augmented_matrix,
            augmented_labels,
            sample_weights(augmented_labels, augmented_records),
            linear,
            steps=2200,
            l2=0.12,
        )
        threshold = threshold_from_real(
            predictor.score(random_conv[train]), train_records
        )
        test_records = [record for record, keep in zip(records, test) if keep]
        augmented_folds.append({
            "held_out_bag": held_out,
            **fold_metrics(
                predictor.score(random_conv[test]), labels[test], test_records, threshold
            ),
        })
    report["models"]["random_conv_domain_randomized"] = summarize(
        augmented_folds, random_conv.shape[1]
    )
    mild_folds = []
    for held_out in sorted(set(groups.tolist())):
        train = groups != held_out
        test = ~train
        train_records = [record for record, keep in zip(records, train) if keep]
        fitting_records = train_records * 3
        fitting_labels = np.tile(labels[train], 3)
        fitting_matrix = np.vstack((
            random_conv[train],
            mild_randomized_conv[0][train],
            mild_randomized_conv[1][train],
        ))
        predictor = fit_logistic_features(
            fitting_matrix,
            fitting_labels,
            sample_weights(fitting_labels, fitting_records),
            linear,
            steps=2200,
            l2=0.12,
        )
        threshold = threshold_from_real(
            predictor.score(random_conv[train]), train_records
        )
        test_records = [record for record, keep in zip(records, test) if keep]
        mild_folds.append({
            "held_out_bag": held_out,
            **fold_metrics(
                predictor.score(random_conv[test]), labels[test], test_records, threshold
            ),
        })
    report["models"]["random_conv_mild_randomized"] = summarize(
        mild_folds, random_conv.shape[1]
    )
    positive_augmented_folds = []
    for held_out in sorted(set(groups.tolist())):
        train = groups != held_out
        test = ~train
        positive = train & (labels == 1)
        train_records = [record for record, keep in zip(records, train) if keep]
        positive_records = [record for record, keep in zip(records, positive) if keep]
        augmented_records = train_records + positive_records * 2
        augmented_labels = np.concatenate((
            labels[train], labels[positive], labels[positive]
        ))
        augmented_matrix = np.vstack((
            random_conv[train],
            randomized_conv[0][positive],
            randomized_conv[1][positive],
        ))
        predictor = fit_logistic_features(
            augmented_matrix,
            augmented_labels,
            sample_weights(augmented_labels, augmented_records),
            linear,
            steps=2200,
            l2=0.12,
        )
        threshold = threshold_from_real(
            predictor.score(random_conv[train]), train_records
        )
        test_records = [record for record, keep in zip(records, test) if keep]
        positive_augmented_folds.append({
            "held_out_bag": held_out,
            **fold_metrics(
                predictor.score(random_conv[test]), labels[test], test_records, threshold
            ),
        })
    report["models"]["random_conv_positive_randomized"] = summarize(
        positive_augmented_folds, random_conv.shape[1]
    )
    if osdar_cache is not None and osdar_cache.exists():
        external = np.load(osdar_cache, allow_pickle=False)
        external_features = random_convolution_descriptor(external["patches"])
        external_records = json.loads(str(external["records_json"]))
        external_folds = []
        for held_out in sorted(set(groups.tolist())):
            train = groups != held_out
            test = ~train
            train_records = [record for record, keep in zip(records, train) if keep]
            added_records = [
                {"source": "external", "scenario": "osdar_" + record["class"]}
                for record in external_records
            ]
            train_matrix = np.vstack((random_conv[train], external_features))
            train_labels = np.concatenate((labels[train], np.ones(len(external_features))))
            fitting_records = train_records + added_records
            predictor = fit_logistic_features(
                train_matrix,
                train_labels,
                sample_weights(train_labels, fitting_records),
                linear,
                steps=2200,
                l2=0.12,
            )
            threshold = threshold_from_real(
                predictor.score(random_conv[train]), train_records
            )
            test_records = [record for record, keep in zip(records, test) if keep]
            external_folds.append({
                "held_out_bag": held_out,
                **fold_metrics(
                    predictor.score(random_conv[test]), labels[test], test_records, threshold
                ),
            })
        report["models"]["random_conv_plus_osdar"] = summarize(
            external_folds, random_conv.shape[1]
        )
        report["external_training_samples"] = len(external_features)
    if weak_cache is not None and weak_cache.exists():
        weak = np.load(weak_cache, allow_pickle=False)
        weak_features = random_convolution_descriptor(weak["patches"])
        weak_records_raw = json.loads(str(weak["records_json"]))
        weak_folds = []
        for held_out in sorted(set(groups.tolist())):
            train = groups != held_out
            test = ~train
            train_records = [record for record, keep in zip(records, train) if keep]
            weak_records = [
                {"source": "weak_real", "scenario": "corridor_memory"}
                for _ in weak_features
            ]
            fitting_matrix = np.vstack((random_conv[train], weak_features))
            fitting_labels = np.concatenate((labels[train], np.ones(len(weak_features))))
            fitting_records = train_records + weak_records
            predictor = fit_logistic_features(
                fitting_matrix,
                fitting_labels,
                sample_weights(fitting_labels, fitting_records),
                linear,
                steps=2200,
                l2=0.12,
            )
            threshold = threshold_from_real(
                predictor.score(random_conv[train]), train_records
            )
            test_records = [record for record, keep in zip(records, test) if keep]
            weak_folds.append({
                "held_out_bag": held_out,
                **fold_metrics(
                    predictor.score(random_conv[test]), labels[test], test_records, threshold
                ),
            })
        report["models"]["random_conv_plus_weak_real"] = summarize(
            weak_folds, random_conv.shape[1]
        )
        report["weak_training_samples"] = len(weak_features)
        hotspot = float(weak["hotspots"][0])
        precise_mask = np.asarray([
            abs(record["world_m"] - hotspot) <= 0.9
            and abs(record["lateral_m"]) <= 0.15
            for record in weak_records_raw
        ])
        precise_features = weak_features[precise_mask]
        precise_folds = []
        for held_out in sorted(set(groups.tolist())):
            train = groups != held_out
            test = ~train
            train_records = [record for record, keep in zip(records, train) if keep]
            precise_records = [
                {"source": "weak_real", "scenario": "central_rail_object"}
                for _ in precise_features
            ]
            fitting_matrix = np.vstack((random_conv[train], precise_features))
            fitting_labels = np.concatenate((labels[train], np.ones(len(precise_features))))
            fitting_records = train_records + precise_records
            predictor = fit_logistic_features(
                fitting_matrix,
                fitting_labels,
                sample_weights(fitting_labels, fitting_records),
                linear,
                steps=2200,
                l2=0.12,
            )
            threshold = threshold_from_real(
                predictor.score(random_conv[train]), train_records
            )
            test_records = [record for record, keep in zip(records, test) if keep]
            precise_folds.append({
                "held_out_bag": held_out,
                **fold_metrics(
                    predictor.score(random_conv[test]), labels[test], test_records, threshold
                ),
            })
        report["models"]["random_conv_plus_precise_weak_real"] = summarize(
            precise_folds, random_conv.shape[1]
        )
        report["precise_weak_training_samples"] = len(precise_features)
        transplant_folds = []
        donor_patches = weak["patches"][precise_mask]
        real_mask = np.asarray([record["source"] == "real" for record in records])
        for fold_index, held_out in enumerate(sorted(set(groups.tolist()))):
            train = groups != held_out
            test = ~train
            train_records = [record for record, keep in zip(records, train) if keep]
            background_mask = train & real_mask
            transplanted = transplant_real_object(
                donor_patches,
                data["patches"][background_mask],
                seed=20261010 + fold_index,
            )
            transplanted_features = random_convolution_descriptor(transplanted)
            transplant_records = [
                {"source": "real_transplant", "scenario": "central_rail_object"}
                for _ in transplanted_features
            ]
            fitting_matrix = np.vstack((random_conv[train], transplanted_features))
            fitting_labels = np.concatenate((
                labels[train], np.ones(len(transplanted_features))
            ))
            fitting_records = train_records + transplant_records
            predictor = fit_logistic_features(
                fitting_matrix,
                fitting_labels,
                sample_weights(fitting_labels, fitting_records),
                linear,
                steps=2200,
                l2=0.12,
            )
            threshold = threshold_from_real(
                predictor.score(random_conv[train]), train_records
            )
            test_records = [record for record, keep in zip(records, test) if keep]
            transplant_folds.append({
                "held_out_bag": held_out,
                "transplanted_training_samples": len(transplanted_features),
                **fold_metrics(
                    predictor.score(random_conv[test]), labels[test], test_records, threshold
                ),
            })
        report["models"]["random_conv_real_transplant"] = summarize(
            transplant_folds, random_conv.shape[1]
        )
        combined_folds = []
        for fold_index, held_out in enumerate(sorted(set(groups.tolist()))):
            train = groups != held_out
            test = ~train
            train_records = [record for record, keep in zip(records, train) if keep]
            background_mask = train & real_mask
            transplanted = transplant_real_object(
                donor_patches,
                data["patches"][background_mask],
                seed=20261020 + fold_index,
            )
            transplanted = randomized_patches(
                transplanted, seed=20261030 + fold_index
            )
            transplanted_features = random_convolution_descriptor(transplanted)
            fitting_matrix = np.vstack((
                random_conv[train],
                randomized_conv[0][train],
                randomized_conv[1][train],
                transplanted_features,
            ))
            fitting_labels = np.concatenate((
                labels[train], labels[train], labels[train],
                np.ones(len(transplanted_features)),
            ))
            fitting_records = train_records * 3 + [
                {"source": "randomized_transplant", "scenario": "central_rail_object"}
                for _ in transplanted_features
            ]
            predictor = fit_logistic_features(
                fitting_matrix,
                fitting_labels,
                sample_weights(fitting_labels, fitting_records),
                linear,
                steps=2400,
                l2=0.12,
            )
            threshold = threshold_from_real(
                predictor.score(random_conv[train]), train_records
            )
            test_records = [record for record, keep in zip(records, test) if keep]
            combined_folds.append({
                "held_out_bag": held_out,
                **fold_metrics(
                    predictor.score(random_conv[test]), labels[test], test_records, threshold
                ),
            })
        report["models"]["random_conv_transplant_plus_randomization"] = summarize(
            combined_folds, random_conv.shape[1]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--weak-cache",
        type=Path,
        default=Path("lidar_geometry/artifacts/weak_obstacle_candidates.npz"),
    )
    parser.add_argument(
        "--osdar-cache",
        type=Path,
        default=Path("lidar_geometry/artifacts/osdar_spatial_cache.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/invariant_spatial_comparison.json"),
    )
    args = parser.parse_args()
    evaluate(args.cache, args.output, args.osdar_cache, args.weak_cache)


if __name__ == "__main__":
    main()

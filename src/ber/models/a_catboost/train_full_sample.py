"""Train Pipeline A on a full-corpus candidate sample with grouped validation."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool

from ber.data.build_normalized import sha256, write_json
from ber.evaluation.entity_score import choose_threshold, decisions, score
from ber.models.a_catboost.data import ROLES, groups, make_splits, validate_splits


def checked_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "PASS" or (root / "_SUCCESS").read_text().strip() != sha256(root / "manifest.json"):
        raise ValueError(f"Incomplete input: {root}")
    for name, digest in manifest.get("files", manifest.get("output_hashes", {})).items():
        if sha256(root / name) != digest:
            raise ValueError(f"Input checksum mismatch: {root / name}")
    return manifest


def load(bundle, features):
    sample = checked_manifest(bundle)
    feature_manifest = checked_manifest(features)
    if feature_manifest.get("bundle_manifest_sha256") != sha256(bundle / "manifest.json"):
        raise ValueError("Feature and sample bundles do not align")
    records = {row["entity_id"]: row for row in pq.read_table(bundle / "queries.parquet").to_pylist()}
    truth = {}
    for row in pq.read_table(bundle / "truth.parquet").to_pylist():
        value = row["matched_entity_ids"]
        truth[row["source1_entity_id"]] = value.split(",") if value else []
    if set(records) != set(truth) or len(truth) != sample["anchors"]:
        raise ValueError("Anchor records and truth differ")
    schema = json.loads((features / "feature_schema.json").read_text())
    names = schema["features"]
    if len(names) != len(set(names)) or set(names) & {"source1_entity_id", "candidate_id", "label"}:
        raise ValueError("Invalid feature schema")
    table = pq.read_table(features / "pair_features.parquet")
    if table.num_rows != sample["candidate_pairs"]:
        raise ValueError("Feature row count differs from candidate bundle")
    pairs = list(zip(table["source1_entity_id"].to_pylist(), table["candidate_id"].to_pylist()))
    if any(anchor not in truth for anchor, _ in pairs):
        raise ValueError("Unknown feature anchor")
    matrix = np.column_stack([table[name].to_numpy() for name in names]).astype(np.float32)
    if np.isinf(matrix).any():
        raise ValueError("Infinite feature value")
    positives = {anchor: set(ids) for anchor, ids in truth.items()}
    labels = np.fromiter((int(target in positives[anchor]) for anchor, target in pairs),
                         dtype=np.int8, count=len(pairs))
    return matrix, labels, names, pairs, truth, records


def compact_score(actual, pairs, probabilities, threshold, records):
    predicted = decisions(actual, pairs, probabilities, threshold)
    result = score(actual, predicted)
    result.pop("per_anchor")
    result["country_slices"] = {}
    for country in sorted({records[anchor]["country"] for anchor in actual}):
        subset = {anchor: ids for anchor, ids in actual.items() if records[anchor]["country"] == country}
        country_score = score(subset, {anchor: predicted[anchor] for anchor in subset})
        country_score.pop("per_anchor")
        result["country_slices"][country] = country_score
    return result


def train(bundle, features, config_path, output, skip_holdout=False):
    if output.exists():
        raise FileExistsError(output)
    started = time.time()
    config = json.loads(config_path.read_text())
    matrix, labels, names, pairs, truth, records = load(bundle, features)
    assignments, group_map = make_splits(truth, records, config["seed"])
    validate_splits(assignments, truth, group_map)
    masks = {role: np.fromiter((assignments[anchor] == role for anchor, _ in pairs),
                               dtype=np.bool_, count=len(pairs)) for role in ROLES}
    for role in ("train", "early_stop"):
        if len(np.unique(labels[masks[role]])) != 2:
            raise ValueError(f"{role} needs both label classes")
    output.mkdir(parents=True)
    write_json(output / "splits.json", assignments)
    write_json(output / "config.json", config)
    model = CatBoostClassifier(
        iterations=config["iterations"], depth=config["depth"],
        learning_rate=config["learning_rate"], l2_leaf_reg=config["l2_leaf_reg"],
        random_seed=config["seed"], thread_count=config["thread_count"], task_type="CPU",
        loss_function="Logloss", eval_metric="Logloss", nan_mode="Min",
        allow_writing_files=False, verbose=100)
    model.fit(Pool(matrix[masks["train"]], labels[masks["train"]], feature_names=names),
              eval_set=Pool(matrix[masks["early_stop"]], labels[masks["early_stop"]], feature_names=names),
              early_stopping_rounds=config["early_stopping_rounds"], use_best_model=True)
    model.save_model(str(output / "model.cbm"))
    write_json(output / "feature_names.json", names)
    probabilities = model.predict_proba(Pool(matrix, feature_names=names))[:, 1]

    def partition(role):
        indices = np.flatnonzero(masks[role])
        actual = {anchor: ids for anchor, ids in truth.items() if assignments[anchor] == role}
        return actual, [pairs[index] for index in indices], probabilities[indices], labels[indices]

    calibration_truth, calibration_pairs, calibration_probs, _ = partition("calibration")
    selected, curve = choose_threshold(calibration_truth, calibration_pairs, calibration_probs)
    write_json(output / "threshold.json", {**selected, "selected_on": "calibration only",
                                               "decision": "probability >= threshold"})
    pq.write_table(pa.Table.from_pylist(curve), output / "threshold_curve.parquet", compression="zstd")
    metrics = {}
    prediction_rows = []
    for role in ROLES:
        if skip_holdout and role == "holdout":
            continue
        actual, role_pairs, role_probs, role_labels = partition(role)
        result = compact_score(actual, role_pairs, role_probs, selected["threshold"], records)
        result["candidate_oracle_macro_f05"] = compact_score(
            actual, role_pairs, role_labels.astype(np.float64), 0.5, records)["macro_f05"]
        result["candidate_pairs"] = len(role_pairs)
        result["positive_candidate_pairs"] = int(role_labels.sum())
        metrics[role] = result
        if role != "train":
            prediction_rows.extend({"source1_entity_id": anchor, "candidate_id": target,
                                    "split": role, "label": int(label), "probability": float(probability)}
                                   for (anchor, target), label, probability in zip(
                                       role_pairs, role_labels, role_probs))
    pq.write_table(pa.Table.from_pylist(prediction_rows), output / "validation_predictions.parquet",
                   compression="zstd")
    write_json(output / "feature_importance.json",
               sorted(({"feature": name, "importance": float(importance)}
                       for name, importance in zip(names, model.feature_importances_)),
                      key=lambda row: -row["importance"]))
    report = {"status": "PASS", "scope": "group-disjoint sampled full-corpus candidates; not leaderboard",
              "holdout_evaluated": not skip_holdout,
              "sample_anchors": len(truth), "sample_pairs": len(pairs),
              "group_count": len(set(group_map.values())),
              "anchor_counts": {role: sum(value == role for value in assignments.values()) for role in ROLES},
              "feature_count": len(names), "best_iteration": model.get_best_iteration(),
              "tree_count": model.tree_count_, "threshold": selected,
              "metrics": metrics,
              "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
              "features_manifest_sha256": sha256(features / "manifest.json"),
              "config_sha256": sha256(config_path),
              "elapsed_seconds": round(time.time() - started, 2)}
    write_json(output / "report.json", report)
    (output / "_SUCCESS").write_text(sha256(output / "report.json") + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-holdout", action="store_true",
                        help="Use for hyperparameter development; leave the holdout unscored")
    args = parser.parse_args()
    report = train(args.bundle, args.features, args.config, args.output, args.skip_holdout)
    summary = {"anchor_counts": report["anchor_counts"],
               "tree_count": report["tree_count"], "threshold": report["threshold"]}
    if not args.skip_holdout:
        summary["holdout_macro_f05"] = report["metrics"]["holdout"]["macro_f05"]
    print(json.dumps(summary, indent=2))

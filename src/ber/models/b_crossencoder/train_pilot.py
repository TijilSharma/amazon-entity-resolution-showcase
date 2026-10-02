"""Train and compare a bounded multilingual cross-encoder on the shared sample."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ber.data.build_normalized import sha256, write_json
from ber.evaluation.entity_score import choose_threshold, decisions, score
from ber.models.a_catboost.train_full_sample import checked_manifest


MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


def selected_anchors(rows, cap, seed):
    anchors = {row["source1_entity_id"] for row in rows}
    return set(sorted(anchors, key=lambda a: hashlib.sha256(f"{seed}:{a}".encode()).digest())[:cap])


def text_pair(row):
    def format_side(prefix):
        return (f"name: {row[prefix + '_name']} ; address: {row[prefix + '_address']} ; "
                f"country: {row[prefix + '_country']}")
    return format_side("query"), format_side("target")


class Pairs(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(tokenizer, length):
    def convert(rows):
        pairs = [text_pair(row) for row in rows]
        encoded = tokenizer([x[0] for x in pairs], [x[1] for x in pairs],
                            padding=True, truncation=True, max_length=length,
                            return_tensors="pt")
        labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
        return encoded, labels, rows
    return convert


def predict(model, loader, device):
    model.eval()
    pairs, probabilities = [], []
    with torch.inference_mode():
        for encoded, _, rows in loader:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**encoded).logits.reshape(-1)
            probabilities.extend(torch.sigmoid(logits.float()).cpu().tolist())
            pairs.extend((row["source1_entity_id"], row["candidate_id"]) for row in rows)
    return pairs, np.array(probabilities, dtype=np.float64)


def truth_for(bundle, anchors):
    truth = {}
    for row in pq.read_table(bundle / "truth.parquet").to_pylist():
        anchor = row["source1_entity_id"]
        if anchor in anchors:
            truth[anchor] = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
    if set(truth) != anchors:
        raise ValueError("Truth and selected anchors differ")
    return truth


def compact(actual, pairs, probabilities, threshold):
    result = score(actual, decisions(actual, pairs, probabilities, threshold))
    result.pop("per_anchor", None)
    return result


def train(input_dir, bundle, compare_a, output, max_train_anchors, max_eval_anchors,
          batch_size, max_steps, max_length, seed):
    if output.exists():
        raise FileExistsError(output)
    checked_manifest(input_dir)
    checked_manifest(bundle)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("The Pipeline B pilot requires the laptop GPU")
    device = torch.device("cuda")
    started = time.time()
    output.mkdir(parents=True)
    data = {}
    counts = {}
    for role in ("train", "early_stop", "calibration"):
        rows = pq.read_table(input_dir / f"{role}.parquet").to_pylist()
        cap = max_train_anchors if role == "train" else max_eval_anchors
        anchors = selected_anchors(rows, cap, seed + {"train": 0, "early_stop": 1, "calibration": 2}[role])
        data[role] = [row for row in rows if row["source1_entity_id"] in anchors]
        counts[role] = {"anchors": len(anchors), "pairs": len(data[role]),
                        "positives": sum(row["label"] for row in data[role])}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=REVISION).to(device)
    if model.config.num_labels != 1:
        raise ValueError("Expected one-logit reranker")
    loaders = {
        role: DataLoader(Pairs(rows), batch_size=batch_size, shuffle=(role == "train"),
                         num_workers=0, collate_fn=collate(tokenizer, max_length))
        for role, rows in data.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    step = 0
    losses = []
    for encoded, labels, _ in loaders["train"]:
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**encoded).logits.reshape(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        losses.append(float(loss.detach().cpu()))
        if step % 100 == 0:
            print(f"B step {step}/{max_steps} loss={np.mean(losses[-100:]):.4f} "
                  f"elapsed={time.time()-started:.1f}s", flush=True)
        if step >= max_steps:
            break
    model.save_pretrained(output / "model", safe_serialization=True)
    tokenizer.save_pretrained(output / "model")
    predictions = {}
    for role in ("early_stop", "calibration"):
        pairs, probs = predict(model, loaders[role], device)
        actual = truth_for(bundle, {row["source1_entity_id"] for row in data[role]})
        predictions[role] = (actual, pairs, probs)
    calibration = predictions["calibration"]
    threshold, _ = choose_threshold(*calibration)
    metrics = {role: compact(*values, threshold["threshold"])
               for role, values in predictions.items()}
    a_threshold = json.loads((compare_a / "threshold.json").read_text())["threshold"]
    a_rows = pq.read_table(compare_a / "validation_predictions.parquet",
                           columns=["source1_entity_id", "candidate_id", "split", "probability"]).to_pylist()
    a_metrics = {}
    a_aligned = {}
    prediction_rows = []
    for role, (actual, b_pairs, _) in predictions.items():
        subset = [row for row in a_rows if row["split"] == role and row["source1_entity_id"] in actual]
        a_pairs = [(row["source1_entity_id"], row["candidate_id"]) for row in subset]
        if len(a_pairs) != len(b_pairs) or set(a_pairs) != set(b_pairs):
            raise ValueError(f"A and B candidate universes differ for {role}")
        a_metrics[role] = compact(actual, a_pairs,
                                  np.array([row["probability"] for row in subset]), a_threshold)
        a_map = dict(zip(a_pairs, (row["probability"] for row in subset)))
        a_aligned[role] = np.array([a_map[pair] for pair in b_pairs], dtype=np.float64)
        b_map = dict(zip(b_pairs, predictions[role][2]))
        prediction_rows.extend({"source1_entity_id": row["source1_entity_id"],
                                "candidate_id": row["candidate_id"], "role": role,
                                "label": row["label"],
                                "B_probability": float(b_map[(row["source1_entity_id"], row["candidate_id"])]),
                                "A_probability": float(a_map[(row["source1_entity_id"], row["candidate_id"])])}
                               for row in data[role])
    pq.write_table(pa.Table.from_pylist(prediction_rows), output / "validation_predictions.parquet",
                   compression="zstd")
    ensemble_trials = []
    for weight_a in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = {role: weight_a * a_aligned[role] + (1 - weight_a) * predictions[role][2]
                   for role in predictions}
        chosen, _ = choose_threshold(predictions["calibration"][0],
                                     predictions["calibration"][1], blended["calibration"])
        ensemble_trials.append({"weight_A": weight_a, "threshold": chosen["threshold"],
                                "calibration": compact(predictions["calibration"][0],
                                                       predictions["calibration"][1],
                                                       blended["calibration"], chosen["threshold"]),
                                "early_stop": compact(predictions["early_stop"][0],
                                                      predictions["early_stop"][1],
                                                      blended["early_stop"], chosen["threshold"])})
    report = {"status": "PASS", "scope": "bounded Pipeline B pilot; sampled candidates; not leaderboard",
              "model_id": MODEL_ID, "revision": REVISION, "license": "apache-2.0",
              "parameters": sum(p.numel() for p in model.parameters()),
              "counts": counts, "steps": step, "batch_size": batch_size,
              "max_length": max_length, "threshold": threshold,
              "metrics": metrics, "baseline_A_metrics": a_metrics,
              "ensemble_trials": ensemble_trials,
              "baseline_A_model": str(compare_a),
              "elapsed_seconds": round(time.time() - started, 2),
              "input_manifest_sha256": sha256(input_dir / "manifest.json")}
    write_json(output / "report.json", report)
    (output / "_SUCCESS").write_text(sha256(output / "report.json") + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/b_anushka/crossencoder-pairs-full-eval-v2"))
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/a_tijil/full-train-sample-v1"))
    parser.add_argument("--compare-a", type=Path, default=Path("artifacts/a_tijil/full-catboost-depth8-dev"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/b_anushka/crossencoder-pilot-v1"))
    parser.add_argument("--max-train-anchors", type=int, default=3000)
    parser.add_argument("--max-eval-anchors", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2109)
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.bundle, args.compare_a, args.output, args.max_train_anchors,
                           args.max_eval_anchors, args.batch_size, args.max_steps,
                           args.max_length, args.seed), indent=2))

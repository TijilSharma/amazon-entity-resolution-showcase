# Pipeline B: text-pair input

`prepare_pairs.py` builds a bounded cross-encoder training set from the same
20,000-anchor full-corpus candidate sample and anchor splits used by Pipeline A.
It keeps all retrieved positives and up to 20 highest-ranked negatives per
anchor. The output contains normalized name/address fields and country for
both records, the candidate rank, and a training label. Entity IDs are for
alignment only; do not put them in the model text. Holdout anchors and French
test labels are excluded.

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m ber.models.b_crossencoder.prepare_pairs \
  --bundle artifacts/a_tijil/full-train-sample-v1 \
  --splits artifacts/a_tijil/full-catboost-v1/splits.json \
  --output artifacts/b_anushka/crossencoder-pairs-v1 \
  --negatives-per-anchor 20
```

Use `train.parquet` for fitting, `early_stop.parquet` to select a checkpoint,
and `calibration.parquet` to select the decision threshold. These files are
model input, not a trained neural network or a score. Training/evaluation code
must preserve the complete candidate set at validation and test time, counting
unretrieved true links as misses and including singleton anchors.

For a fair A/B comparison, the `--full-eval` option retains every retrieved
candidate for early-stop and calibration anchors while keeping the training
negatives capped at 20. The resulting artifact is
`artifacts/b_anushka/crossencoder-pairs-full-eval-v2`. It has 276,826 training
pairs and about 287,000 pairs in each evaluation role.

`train_pilot.py` fine-tunes the Apache-2.0 multilingual MiniLM cross-encoder
[`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
at pinned revision `1427fd652930e4ba29e8149678df786c240d8825`. The model
has 117,641,089 parameters. It uses name, address and country text from each
record; no entity IDs or retrieval rank enter the neural model. It runs on
Tijil's RTX 4060 laptop GPU while Pipeline A scores on CPU.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  .venv/bin/python -m ber.models.b_crossencoder.train_pilot \
  --input artifacts/b_anushka/crossencoder-pairs-full-eval-v2 \
  --output artifacts/b_anushka/crossencoder-full-epoch-v2 \
  --max-train-anchors 12000 --max-eval-anchors 500 \
  --batch-size 16 --max-steps 18000 --max-length 128
```

The script selects evaluation anchors deterministically, evaluates every
candidate for those anchors, counts unretrieved true links, and compares with
the depth-8 CatBoost model on exactly the same pairs. It writes an A/B report,
prediction file, and simple probability-blend trials. These are development
results; Pipeline B test inference and a final submission are not yet built.

## Full-epoch development comparison (2026-09-26)

`artifacts/b_anushka/crossencoder-full-epoch-v2/report.json` passed. Training
used 12,000 anchors, 276,826 pairs, and one complete epoch (17,302 optimizer
steps) in 1,426 seconds on the RTX 4060. Each evaluation role uses 500
deterministically selected anchors and **all** their retrieved candidates.
The same candidate pairs and anchors were scored by depth-8 CatBoost A.

| Model | Early-stop macro F0.5 | Calibration macro F0.5 |
| --- | ---: | ---: |
| CatBoost A | 0.909640 | 0.893513 |
| Cross-encoder B | 0.929576 | 0.905915 |
| 50% A / 50% B probability blend | 0.930977 | 0.908959 |

The blend weight and threshold are exploratory trials; the threshold for each
trial was chosen on calibration anchors. These are sampled US/India
development results, **not** French test or leaderboard scores. Neither model
can recover a true link absent from the retrieved candidate set. Before using
B for submission, build French test inference and validate the resulting TSVs.

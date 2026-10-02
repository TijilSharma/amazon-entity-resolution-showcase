# Reproduction notes

The source and synthetic tests are public; the organizer's records, labels, generated candidates, trained weights, and final predictions are **not**. This repository supports code review and component-level testing without the private dataset. It does not claim that a visitor can recreate the scored submission from public files alone.

Use Linux x86_64 and Python 3.11.11. The challenge environment used CUDA-enabled PyTorch 2.6.0 and an RTX 4060 laptop. Install `requirements-training.lock` using the command in the root README, then run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

If you are an authorized challenge participant with the original `student_resource/dataset` directory, the upstream stages start as follows:

```bash
export PYTHONPATH=src
RAW_DATASET=/absolute/path/to/student_resource/dataset
.venv/bin/python -m ber.data.audit_dataset \
  --data-root "$RAW_DATASET" --report manifests/input_audit.json
.venv/bin/python -m ber.data.build_normalized \
  --data-root "$RAW_DATASET" --audit manifests/input_audit.json \
  --output data/processed/conservative-v1-py311-u14 --workers 3
.venv/bin/python -m ber.retrieval.full_exact \
  --release data/processed/conservative-v1-py311-u14 --split test \
  --output candidates/full-exact-v1/test --cap 50
.venv/bin/python -m ber.retrieval.full_fuzzy \
  --release data/processed/conservative-v1-py311-u14 --split test \
  --output candidates/full-fuzzy-selective-v2/test --cap 50 --workers 4
.venv/bin/python -m ber.retrieval.full_dense \
  --release data/processed/conservative-v1-py311-u14 --split test \
  --output candidates/full-e5-v1/test --device cuda \
  --k 50 --nlist 4096 --nprobe 16
.venv/bin/python -m ber.retrieval.union_full \
  --split test --inputs candidates/full-exact-v1/test \
  candidates/full-fuzzy-selective-v2/test candidates/full-e5-v1/test \
  --output candidates/full-union-v1/test --cap 100
```

The model modules under `src/ber/models/` contain training, scoring, specialist rescue, conflict resolution, and final filtering. Their saved checkpoints are omitted here. Exact full-corpus reproduction requires those checkpoints or retraining and may differ at floating-point/FAISS ties. There are no organizer files, public prediction samples, or external business lookups in this repository.

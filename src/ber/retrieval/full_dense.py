"""Batch full-corpus E5 retrieval using a compressed FAISS IVF-PQ index.

The index is trained only on provided target records. It never calls an
external business service. Output is route provenance, not final matches.
"""
import argparse
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from ber.data.build_normalized import sha256, write_json
from ber.retrieval.run_pilot import MODEL_ID, REVISION

VERSION = "full-e5-ivfpq-v1"
COLUMNS = ["entity_id", "name_normalized", "address_normalized"]


def batches(release, split, source, size=128):
    buffer = []
    for path in sorted((release / split / source).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS, batch_size=50_000):
            for row in batch.to_pylist():
                buffer.append(row)
                if len(buffer) == size:
                    yield buffer
                    buffer = []
    if buffer:
        yield buffer


def text(row):
    return " | ".join(value for value in (row["name_normalized"], row["address_normalized"]) if value)


def model(cache, device):
    local = os.environ.get("BER_E5_LOCAL_MODEL")
    if local:
        snapshot = local
    else:
        patterns = ["config.json", "model.safetensors", "modules.json", "1_Pooling/config.json",
                    "sentence_bert_config.json", "sentencepiece.bpe.model", "special_tokens_map.json",
                    "tokenizer.json", "tokenizer_config.json", "README.md"]
        snapshot = snapshot_download(MODEL_ID, revision=REVISION, cache_dir=cache, allow_patterns=patterns)
    result = SentenceTransformer(snapshot, device=device, local_files_only=True)
    result.max_seq_length = 128
    return result


def encode(encoder, rows):
    values = [text(row) for row in rows]
    vectors = np.zeros((len(rows), encoder.get_sentence_embedding_dimension()), dtype=np.float32)
    present = [i for i, value in enumerate(values) if value]
    if present:
        vectors[present] = encoder.encode(["query: " + values[i] for i in present], batch_size=len(rows),
                                           normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vectors


def total_targets(release, split):
    manifest = json.loads((release / "manifest.json").read_text())
    return sum(manifest["sources"][f"{split}_{source}.tsv"]["rows"] for source in ("source2", "source3"))


def run(release, split, output, cache, device, k, nlist, nprobe, resume=False):
    if output.exists() and not resume:
        raise FileExistsError(f"Output exists: {output}; pass --resume only for an interrupted BUILDING run")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    release_hash = sha256(release / "manifest.json")
    manifest = {"status": "BUILDING", "version": VERSION, "split": split, "model": MODEL_ID,
                "revision": REVISION, "k": k, "nlist": nlist, "nprobe": nprobe,
                "release_manifest_sha256": release_hash, "device": device}
    if resume and (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        contents = [p.name for p in output.iterdir() if p.name != "manifest.json"]
        if previous.get("status") != "BUILDING" or previous.get("release_manifest_sha256") != release_hash or contents:
            raise ValueError("Dense resume is safe only before any index/vector artifact was written")
    write_json(output / "manifest.json", manifest)
    try:
        encoder = model(cache, device)
        sample = []
        for source in ("source2", "source3"):
            for batch in batches(release, split, source):
                sample.append(encode(encoder, batch))
                if sum(x.shape[0] for x in sample) >= 200_000:
                    break
            if sum(x.shape[0] for x in sample) >= 200_000:
                break
        train_vectors = np.concatenate(sample, axis=0)[:200_000]
        dimension = train_vectors.shape[1]
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 48, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(train_vectors)
        capacity = total_targets(release, split)
        identifiers = np.memmap(output / "target_ids.bin", dtype="S20", mode="w+", shape=(capacity,))
        position = 0
        for source in ("source2", "source3"):
            for batch in batches(release, split, source):
                vectors = encode(encoder, batch)
                ids = np.arange(position, position + len(batch), dtype=np.int64)
                index.add_with_ids(vectors, ids)
                identifiers[position:position+len(batch)] = [r["entity_id"].encode() for r in batch]
                position += len(batch)
                if position % 100_000 == 0:
                    print(f"{split}: indexed {position:,}/{capacity:,} targets", flush=True)
        identifiers.flush()
        if position != capacity or index.ntotal != capacity:
            raise ValueError("Indexed target count differs from manifest")
        index.nprobe = nprobe
        faiss.write_index(index, str(output / "index.faiss"))
        total_queries = total_pairs = 0
        for source in ("source1",):
            for shard, path in enumerate(sorted((release / split / source).glob("*.parquet"))):
                emitted = []
                for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS, batch_size=128):
                    query = batch.to_pylist()
                    vectors = encode(encoder, query)
                    scores, matches = index.search(vectors, k)
                    for row, candidate_ids, candidate_scores in zip(query, matches, scores):
                        for rank, (candidate, score) in enumerate(zip(candidate_ids, candidate_scores), 1):
                            if candidate >= 0:
                                emitted.append({"source1_entity_id": row["entity_id"],
                                                "candidate_id": identifiers[candidate].decode(),
                                                "route": "e5_name_address_ivfpq", "route_rank": rank,
                                                "score": float(score)})
                    total_queries += len(query)
                pq.write_table(pa.Table.from_pylist(emitted, schema=pa.schema([
                    ("source1_entity_id", pa.string()), ("candidate_id", pa.string()),
                    ("route", pa.string()), ("route_rank", pa.int32()), ("score", pa.float32())])),
                    output / f"part-{shard:05d}.parquet", compression="zstd")
                total_pairs += len(emitted)
                if shard % 25 == 0:
                    print(f"{split}: searched {total_queries:,} anchors; {total_pairs:,} dense candidates", flush=True)
        manifest.update({"status": "PASS", "targets": capacity, "anchors": total_queries,
                         "candidate_pairs": total_pairs, "index_sha256": sha256(output / "index.faiss"),
                         "target_ids_sha256": sha256(output / "target_ids.bin"),
                         "elapsed_seconds": round(time.time()-started, 2)})
        write_json(output / "manifest.json", manifest)
        (output / "_SUCCESS").write_text(sha256(output / "manifest.json") + "\n")
    except BaseException as error:
        manifest.update({"status": "FAILED", "error": repr(error)})
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", default="cache/huggingface")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.release, args.split, args.output, args.cache, args.device, args.k, args.nlist, args.nprobe, args.resume)

# Methodology

## Task

Given business names, addresses, and country labels from three independent sources, predict which Source-2 and Source-3 records refer to each Source-1 business. Names and addresses can be missing, abbreviated, transliterated, reordered, or written in different scripts. Training labels cover the US and India; the test data also includes France. Source-1 records require an output even when no match exists.

The metric is macro F0.5, so false merges are more costly than missed links. We therefore treated **candidate retrieval**, **pair classification**, and **final assignment** as separate problems.

## 1. Normalize without erasing evidence

`src/ber/data/` preserves each original field and creates auxiliary comparison views. Unicode and whitespace cleanup, accent folding, case normalization, and punctuation handling help retrieval while retaining the raw value for audit. Indic combining marks and leading zeros in numeric tokens stay meaningful. Missing values receive explicit flags; two missing addresses are not counted as address agreement.

## 2. Generate candidates with complementary routes

| Route | Purpose | Main limitation |
|---|---|---|
| Exact name/address keys | High-precision obvious links | Misses spelling and script variation |
| Fuzzy lexical index | Typos, reordered tokens, partial addresses, numbers | Weak across unrelated scripts |
| Multilingual E5 + FAISS IVF-PQ | Cross-script semantic retrieval at scale | Can return plausible but wrong branches |

The routes are deduplicated and merged with provenance. The baseline cap is 100 targets per anchor to bound feature computation and CatBoost scoring. We measured recall against labeled training links before tuning the classifier. The capped union retained 88.90% of true links overall, but only 78.29% in India. That result made India candidate rescue more valuable than simply increasing classifier complexity.

The rescue stage uses separate name and address searches, rare lexical and numeric postings, transliteration, and aliases learned from **training-only** matches. It targets anchors with zero, one to two, or three-plus existing predictions. France was handled as an open-set country; no France labels were available for tuning.

## 3. Score candidate pairs

The CatBoost baseline uses 67 explicit features: normalized name/address similarities, token overlap, edit ratios, number agreement/conflict, script and missingness flags, TF-IDF and multilingual embedding similarities, and retrieval-route evidence. Gradient-boosted trees work well on these mixed tabular signals and provide a fast baseline across a very large candidate set.

Multilingual MiniLM cross-encoders then focus on hard candidates rather than scoring the full Cartesian product. A specialist model was fine-tuned on India rescue examples, and a separate top-10 specialist checked strong older candidates. A BGE reranker supplied a second opinion on selected India links. All task-specific fine-tuning used the challenge's labeled training data; no external business database or test labels were used.

## 4. Calibrate decisions and resolve conflicts

Thresholds were chosen on grouped development splits using macro F0.5. Grouping reduces leakage from related entities across training and evaluation. After pair scoring, a target-ownership step selects one Source-1 owner when multiple anchors claim the same Source-2/3 record. This does **not** force one-to-one matching: an anchor can still own multiple target records.

Final checks require every Source-1 anchor to appear exactly once, no duplicate IDs in an output list, and every selected link to be present in that anchor's candidate set.

## Limits

The public leaderboard score does not establish the private score. Candidate-only oracle values are upper bounds for a particular candidate pool, not trained-model results. France has no labeled development slice. The full run depends on private challenge files and compute-heavy candidate/score artifacts, so this public repository provides source and synthetic tests rather than redistributing the data or frozen output TSVs.

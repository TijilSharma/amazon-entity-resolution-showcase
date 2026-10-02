# Results and interpretation

The final Team Tara submission displayed a **public leaderboard macro F0.5 of 0.949**. I observed an approximate standing of **2,000 among roughly 11,000 teams**. Both numbers are reported from the challenge run; this repository does not claim a private leaderboard score or an officially certified final rank.

The table below uses the full labeled **training** corpus to evaluate the earlier capped retrieval union. It measures whether the correct link was available for the classifier to consider.

| Training candidate measure | Value |
|---|---:|
| Source-1 anchors evaluated | 2,206,821 |
| Labeled positive links | 7,638,365 |
| Exact + E5 + fuzzy true-link recall, capped at 100 | 88.90% |
| US true-link recall | 95.98% |
| India true-link recall | 78.29% |
| Candidate-only oracle macro F0.5 | 0.9496 |

The oracle assumes perfect decisions among **that candidate set** and correct empty outputs. It is not a CatBoost score, a test estimate, or the public leaderboard score. Later India rescue routes expanded the final candidate pool.

The strongest practical lesson was that high-quality pair models could not repair absent candidates. India recall was the clearest bottleneck. Neural reranking and target-ownership constraints then helped precision once those hard candidates were present.

Validation included 41 synthetic-data unit tests, full candidate-subset checks, and the organizer's submission-format validator. The last two were run on private generated TSVs and are not shipped in this public portfolio.

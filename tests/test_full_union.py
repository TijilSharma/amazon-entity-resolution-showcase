import unittest

from ber.retrieval.union_full import merge


def row(route, rank, candidate, score=1.0):
    return {"route": route, "route_rank": rank, "candidate_id": candidate, "score": score}


class FullCandidateUnionTests(unittest.TestCase):
    def test_round_robin_preserves_each_route_before_second_ranks(self):
        values = {"exact": [row("exact", 1, "a"), row("exact", 2, "b")],
                  "dense": [row("dense", 1, "c"), row("dense", 2, "d")]}
        self.assertEqual([candidate for candidate, _ in merge(values, 4)], ["c", "a", "d", "b"])

    def test_duplicate_candidate_keeps_all_route_evidence(self):
        values = {"exact": [row("exact", 1, "a")], "dense": [row("dense", 1, "a"), row("dense", 2, "b")]}
        candidates = merge(values, 2)
        self.assertEqual([candidate for candidate, _ in candidates], ["a", "b"])
        self.assertEqual({e["route"] for e in candidates[0][1]}, {"exact", "dense"})

    def test_cap_is_applied_after_deduplication(self):
        values = {"x": [row("x", 1, "a"), row("x", 2, "b")], "y": [row("y", 1, "a"), row("y", 2, "c")]}
        self.assertEqual([candidate for candidate, _ in merge(values, 2)], ["a", "b"])

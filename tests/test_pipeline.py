"""
tests/test_pipeline.py — Test suite for Adversarial Claim Scoring Engine

Run: python -m pytest tests/ -v
Or individual tests: python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np


# ─────────────────────────────────────────────
# TEST: config
# ─────────────────────────────────────────────
class TestConfig(unittest.TestCase):
    def test_runtime_config_defaults(self):
        from config import RuntimeConfig, WEIGHTS, WEIGHT_SUM
        cfg = RuntimeConfig()
        self.assertEqual(cfg.chunk_size, 800)
        self.assertEqual(cfg.dbscan_eps, 0.35)
        self.assertAlmostEqual(cfg.filter_percent, 0.25)
        self.assertEqual(cfg.weights, WEIGHTS)
        self.assertAlmostEqual(cfg.weight_sum, WEIGHT_SUM)

    def test_runtime_config_override(self):
        from config import RuntimeConfig
        cfg = RuntimeConfig(chunk_size=500, weights=[7, 2, 1])
        self.assertEqual(cfg.chunk_size, 500)
        self.assertEqual(cfg.weights, [7, 2, 1])
        self.assertAlmostEqual(cfg.weight_sum, 10.0)


# ─────────────────────────────────────────────
# TEST: filtering (no API calls)
# ─────────────────────────────────────────────
class TestFiltering(unittest.TestCase):
    def setUp(self):
        from config import RuntimeConfig
        self.cfg = RuntimeConfig(filter_percent=0.5, max_claims=10)

    def test_specificity_score_numbers(self):
        from filtering import specificity_score
        vague_score   = specificity_score("Things improved a lot.")
        specific_score = specificity_score(
            "GDP grew by 8.4% in FY2024 according to the Ministry of Finance."
        )
        self.assertGreater(specific_score, vague_score)

    def test_specificity_score_range(self):
        from filtering import specificity_score
        for claim in [
            "Roads built.",
            "The road network expanded by 47,000 km between 2014–2024.",
            "India's GDP at Rs. 3.7 trillion in 2023.",
        ]:
            score = specificity_score(claim)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_filter_top_percent(self):
        from filtering import filter_top_percent
        claims = [
            "Things improved a lot.",
            "Road network expanded by 47,000 km between 2014–2024.",
            "GDP grew 8.4% in FY2024.",
            "Some growth was recorded.",
        ]
        result = filter_top_percent(claims, self.cfg)
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), len(claims))
        # Check sorted descending
        scores = [s for _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_check_imbalance_empty(self):
        from filtering import check_imbalance
        reason = check_imbalance([], ["claim1"], self.cfg)
        self.assertIsNotNone(reason)

    def test_check_imbalance_extreme(self):
        from filtering import check_imbalance
        aff = ["c"] * 50
        neg = ["c"] * 2
        reason = check_imbalance(aff, neg, self.cfg)
        self.assertIsNotNone(reason)
        self.assertIn("imbalance", reason.lower())

    def test_check_no_imbalance(self):
        from filtering import check_imbalance
        aff = ["c"] * 5
        neg = ["c"] * 5
        reason = check_imbalance(aff, neg, self.cfg)
        self.assertIsNone(reason)

    def test_balance_sides(self):
        from filtering import balance_sides
        aff = ["a"] * 10
        neg = ["n"] * 7
        a, n = balance_sides(aff, neg, self.cfg)
        self.assertEqual(len(a), len(n))
        self.assertLessEqual(len(a), self.cfg.max_claims)


# ─────────────────────────────────────────────
# TEST: clustering math (no API calls)
# ─────────────────────────────────────────────
class TestClustering(unittest.TestCase):
    def _make_vectors(self, n: int, dim: int = 8) -> np.ndarray:
        rng = np.random.default_rng(42)
        v = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / norms

    def test_dbscan_labels(self):
        from clustering import _cluster_dbscan
        vectors = self._make_vectors(20)
        labels = _cluster_dbscan(vectors, eps=0.5, min_samples=2)
        self.assertEqual(len(labels), 20)
        self.assertTrue(all(l >= 0 for l in labels))  # no -1 after reassignment

    def test_kmeans_labels(self):
        from clustering import _cluster_kmeans
        vectors = self._make_vectors(20)
        labels = _cluster_kmeans(vectors, n_clusters=5)
        self.assertEqual(len(labels), 20)
        self.assertEqual(len(set(labels)), 5)

    def test_merge_reduces_count(self):
        from clustering import merge_claims
        from config import RuntimeConfig
        # Craft two groups of very similar vectors
        v1 = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)).astype(np.float32)
        v2 = np.tile([0.0, 1.0, 0.0, 0.0], (5, 1)).astype(np.float32)
        # small perturbation
        rng = np.random.default_rng(0)
        v1 += rng.standard_normal((5, 4)).astype(np.float32) * 0.01
        v2 += rng.standard_normal((5, 4)).astype(np.float32) * 0.01
        vectors = np.vstack([v1, v2])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        claims = [f"Claim group A #{i}" for i in range(5)] + \
                 [f"Claim group B #{i}" for i in range(5)]
        cfg = RuntimeConfig(dbscan_eps=0.2, dbscan_min_samples=2)
        merged = merge_claims(claims, vectors, cfg)
        # Should merge 10 → 2 (one per group)
        self.assertLessEqual(len(merged), 5)
        self.assertGreater(len(merged), 0)


# ─────────────────────────────────────────────
# TEST: aggregation math (no API calls)
# ─────────────────────────────────────────────
class TestAggregation(unittest.TestCase):
    def _make_scored_claims(self, n: int, s1=0.8, s2=0.7, s3=0.6):
        return [{"claim": f"Claim {i}", "s1": s1, "s2": s2, "s3": s3}
                for i in range(n)]

    def test_composite_score(self):
        from aggregation import composite_score
        from config import RuntimeConfig
        cfg = RuntimeConfig(weights=[5, 3, 1])
        s = {"s1": 1.0, "s2": 1.0, "s3": 1.0}
        self.assertAlmostEqual(composite_score(s, cfg), 1.0)
        s_zero = {"s1": 0.0, "s2": 0.0, "s3": 0.0}
        self.assertAlmostEqual(composite_score(s_zero, cfg), 0.0)

    def test_affirmative_wins(self):
        from aggregation import aggregate
        from config import RuntimeConfig
        cfg = RuntimeConfig(conclusion_thresh=0.05, imbalance_ratio=10.0)
        aff = self._make_scored_claims(5, s1=0.9, s2=0.8, s3=0.7)
        neg = self._make_scored_claims(5, s1=0.3, s2=0.2, s3=0.1)
        result = aggregate(aff, neg, cfg)
        self.assertEqual(result["verdict"], "affirmative")
        self.assertGreater(result["avg_aff"], result["avg_neg"])
        self.assertGreater(result["confidence"], 0)

    def test_negative_wins(self):
        from aggregation import aggregate
        from config import RuntimeConfig
        cfg = RuntimeConfig(conclusion_thresh=0.05, imbalance_ratio=10.0)
        aff = self._make_scored_claims(5, s1=0.2, s2=0.2, s3=0.1)
        neg = self._make_scored_claims(5, s1=0.9, s2=0.9, s3=0.8)
        result = aggregate(aff, neg, cfg)
        self.assertEqual(result["verdict"], "negative")

    def test_no_conclusion_gap_too_small(self):
        from aggregation import aggregate
        from config import RuntimeConfig
        cfg = RuntimeConfig(conclusion_thresh=0.5)  # very high threshold
        aff = self._make_scored_claims(5, s1=0.7, s2=0.6, s3=0.5)
        neg = self._make_scored_claims(5, s1=0.7, s2=0.6, s3=0.5)
        result = aggregate(aff, neg, cfg)
        self.assertEqual(result["verdict"], "no_conclusion")

    def test_no_conclusion_imbalance(self):
        from aggregation import aggregate
        from config import RuntimeConfig
        cfg = RuntimeConfig(imbalance_ratio=3.0)
        aff = self._make_scored_claims(30)
        neg = self._make_scored_claims(3)
        result = aggregate(aff, neg, cfg)
        self.assertEqual(result["verdict"], "no_conclusion")
        self.assertIn("imbalance", result["no_conclusion_reason"].lower())

    def test_no_conclusion_empty_side(self):
        from aggregation import aggregate
        from config import RuntimeConfig
        cfg = RuntimeConfig()
        result = aggregate([], self._make_scored_claims(5), cfg)
        self.assertEqual(result["verdict"], "no_conclusion")


# ─────────────────────────────────────────────
# TEST: claim parser in agents (no API)
# ─────────────────────────────────────────────
class TestAgentParser(unittest.TestCase):
    def test_parse_numbered_claims(self):
        from agents import _parse_claims
        raw = """1. Road network expanded by 47,000 km between 2014–2024.
2. GDP grew at 7.2% annually over the decade.
3. Vague.
4. Solar capacity reached 70 GW by 2023.
"""
        claims = _parse_claims(raw)
        self.assertEqual(len(claims), 3)  # "Vague." is too short
        self.assertIn("Road network expanded", claims[0])

    def test_parse_empty(self):
        from agents import _parse_claims
        self.assertEqual(_parse_claims(""), [])

    def test_parse_rejects_short(self):
        from agents import _parse_claims
        raw = "1. Good.\n2. Bad.\n3. This is long enough to pass the filter easily."
        claims = _parse_claims(raw)
        self.assertEqual(len(claims), 1)


# ─────────────────────────────────────────────
# EXPECTED OUTPUT FORMAT (documentation test)
# ─────────────────────────────────────────────
class TestExpectedOutputFormat(unittest.TestCase):
    """Documents the expected output shape of each layer."""

    def test_agg_result_keys(self):
        from aggregation import aggregate
        from config import RuntimeConfig
        cfg = RuntimeConfig(conclusion_thresh=0.01)
        aff = [{"claim": "c", "s1": 0.8, "s2": 0.7, "s3": 0.6}]
        neg = [{"claim": "c", "s1": 0.4, "s2": 0.3, "s3": 0.2}]
        result = aggregate(aff, neg, cfg)
        required_keys = {"verdict", "confidence", "avg_aff", "avg_neg",
                         "gap", "aff_top", "neg_top", "aff_all", "neg_all",
                         "no_conclusion_reason"}
        self.assertTrue(required_keys.issubset(set(result.keys())))

    def test_verdict_values(self):
        from aggregation import aggregate
        from config import RuntimeConfig
        # Verify verdict is one of the three valid values
        cfg = RuntimeConfig(conclusion_thresh=0.01)
        aff = [{"claim": "c", "s1": 0.9, "s2": 0.8, "s3": 0.7}]
        neg = [{"claim": "c", "s1": 0.1, "s2": 0.1, "s3": 0.1}]
        result = aggregate(aff, neg, cfg)
        self.assertIn(result["verdict"], {"affirmative", "negative", "no_conclusion"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

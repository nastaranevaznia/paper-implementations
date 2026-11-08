import os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frlor.config import Config
from frlor.channel import friis_ref_power, link_probability, median_range
from frlor.fuzzy import MamdaniFIS, trimf
from frlor.pipeline import run_once
from frlor import export


def pure_python_fout(fis, nbr, dist, link):
    """Independent, loop-based Mamdani (min/min/max, centroid) used as a reference for the vectorised code."""
    mu = lambda abc, x: float(trimf(np.array([x]), *abc)[0])
    ys = np.linspace(0, 1, fis.cfg.n_points)
    agg = np.zeros_like(ys)
    for rn, rd, rl, ro, w in fis.rules:
        s = w * min(mu(fis.mf["nbr"][rn], nbr), mu(fis.mf["dist"][rd], dist), mu(fis.mf["link"][rl], link))
        for i, y in enumerate(ys):
            agg[i] = max(agg[i], min(s, mu(fis.mf["out"][ro], y)))
    return float(np.trapz(agg * ys, ys) / np.trapz(agg, ys)) if hasattr(np, "trapz") else \
        float(np.trapezoid(agg * ys, ys) / np.trapezoid(agg, ys))


class TestFuzzy(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.fis = MamdaniFIS(self.cfg.fuzzy)

    def test_rule_base_is_complete(self):
        self.assertEqual(len(self.fis.rules), 27)
        self.assertEqual(len({r[:3] for r in self.fis.rules}), 27)

    def test_membership_anchors(self):          # values follow directly from the triangles in Sec. 3.3
        self.assertAlmostEqual(trimf([0], 0, 0, 5)[0], 1.0)
        self.assertAlmostEqual(trimf([5], 3, 7, 11)[0], 0.5)
        self.assertAlmostEqual(trimf([20], 9, 20, 20)[0], 1.0)
        self.assertAlmostEqual(trimf([0.8], 0.6, 1, 1)[0], 0.5)
        self.assertAlmostEqual(trimf([750], 350, 750, 1000)[0], 1.0)

    def test_vectorised_matches_reference(self):
        for x in [(3, 100, 0.9), (7, 600, 0.5), (15, 1000, 0.3), (1, 1200, 0.05), (12, 400, 0.65)]:
            self.assertAlmostEqual(self.fis.evaluate(*[[v] for v in x])[0], pure_python_fout(self.fis, *x), places=6)

    def test_monotone_sense(self):
        good = self.fis.evaluate([5], [50], [0.95])[0]
        bad = self.fis.evaluate([15], [1100], [0.1])[0]
        self.assertGreater(good, 0.8)
        self.assertLess(bad, 0.2)
        self.assertTrue(0.0 <= bad <= good <= 1.0)

    def test_figure_profile_loads(self):
        c = Config(); c.fuzzy.profile = "figure"
        self.assertEqual(MamdaniFIS(c.fuzzy).evaluate([5], [50], [0.95]).shape, (1,))


class TestChannel(unittest.TestCase):
    def test_numbers(self):
        c = Config().channel
        self.assertAlmostEqual(friis_ref_power(c), 1.9228e-4, delta=2e-7)      # Friis @ d0 = 1 m
        r = median_range(c)
        self.assertAlmostEqual(float(link_probability(r, c)), 0.5, places=6)
        d = np.linspace(1, 600, 50)
        self.assertTrue(np.all(np.diff(link_probability(d, c)) <= 1e-12))


class TestReproducibility(unittest.TestCase):
    def cfg(self, seed):
        c = Config(seed=seed)
        c.topology.n_nodes, c.rl.episodes, c.evaluation.n_packets = 150, 80, 100
        return c

    def test_same_seed_same_files(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
            r1 = run_once(self.cfg(3), out_dir=a); r2 = run_once(self.cfg(3), out_dir=b); run_once(self.cfg(4), out_dir=c)
            for m in ("frlor", "fuzzy_only", "por"):
                self.assertEqual(export.sha256(f"{a}/{m}/candidates.txt"), export.sha256(f"{b}/{m}/candidates.txt"))
                self.assertEqual(r1[m]["ent_analytic"], r2[m]["ent_analytic"])
            self.assertEqual(export.sha256(f"{a}/frlor/qtable.json"), export.sha256(f"{b}/frlor/qtable.json"))
            self.assertNotEqual(export.sha256(f"{a}/topology.txt"), export.sha256(f"{c}/topology.txt"))

    def test_candidate_lists_valid(self):
        with tempfile.TemporaryDirectory() as a:
            run_once(self.cfg(5), out_dir=a)
            for line in open(f"{a}/frlor/candidates.txt"):
                if line.startswith("#"):
                    continue
                t = [int(x) for x in line.split()]
                self.assertLessEqual(len(t) - 1, 3)
                self.assertEqual(len(set(t[1:])), len(t) - 1)
                self.assertNotIn(t[0], t[1:])


if __name__ == "__main__":
    unittest.main(verbosity=2)

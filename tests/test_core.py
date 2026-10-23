"""Unit tests for the deterministic core (run: python -m unittest discover -s tests)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from padelcoach import config as C  # noqa: E402
from padelcoach.ball import clean_ball  # noqa: E402
from padelcoach.court import CourtCalibration, lateral_of, wall_region, zone_of  # noqa: E402
from padelcoach.scoreboard import parse_ocr_tokens, point_changes  # noqa: E402
from padelcoach.scoring import PadelScorer  # noqa: E402
from padelcoach.shotfeatures import N_FEAT, sequence_features  # noqa: E402


class ScoringTest(unittest.TestCase):
    def test_game_and_deuce_advantage(self):
        s = PadelScorer(golden_point=False)
        for _ in range(3):
            s.award(0)
            s.award(1)
        self.assertEqual(s.state.point_label(), "Deuce")
        s.award(0)
        self.assertEqual(s.state.point_label(), "Adv A")
        s.award(1)
        self.assertEqual(s.state.point_label(), "Deuce")
        s.award(1)
        info = s.award(1)
        self.assertEqual(info["game_won"], "B")
        self.assertEqual(s.state.games(), [0, 1])

    def test_golden_point(self):
        s = PadelScorer(golden_point=True)
        for _ in range(3):
            s.award(0)
            s.award(1)
        info = s.award(1)
        self.assertEqual(info["game_won"], "B")

    def test_set_and_tiebreak(self):
        s = PadelScorer()
        for g in range(12):  # 6-6
            for _ in range(4):
                s.award(g % 2)
        self.assertTrue(s.state.tiebreak)
        for _ in range(7):
            info = s.award(0)
        self.assertEqual(info["set_won"], "A")
        self.assertEqual(s.state.sets[0], [7, 6])

    def test_server_alternates_each_game(self):
        s = PadelScorer(first_server=0)
        for _ in range(4):
            s.award(0)
        self.assertEqual(s.state.server, 1)

    def test_changeover_after_odd_games(self):
        s = PadelScorer()
        for _ in range(3):
            info = s.award(0)
        info = s.award(0)
        self.assertTrue(info["changeover"])  # after game 1


class CourtTest(unittest.TestCase):
    def setUp(self):
        self.c = CourtCalibration.padeltracker_broadcast()

    def test_calibration_accuracy(self):
        self.assertLess(self.c.reprojection_error_m, 0.05)

    def test_roundtrip(self):
        pts = np.array([[1.0, 2.0], [5.0, 10.0], [9.0, 18.0]])
        back = self.c.to_court(self.c.to_image(pts))
        self.assertTrue(np.allclose(pts, back, atol=1e-3))

    def test_net_projects_to_y10(self):
        p = self.c.to_court(np.array([[960.0, 522.4]]))[0]
        self.assertAlmostEqual(p[1], 10.0, delta=0.15)

    def test_zones(self):
        self.assertEqual(zone_of(10 + 3.5), "net")
        self.assertEqual(zone_of(10 - 5.5), "transition")
        self.assertEqual(zone_of(19.0), "defense")

    def test_lateral_perspective(self):
        # near player sees image-left as their left; far player the opposite
        self.assertEqual(lateral_of(2.0, "near"), "left")
        self.assertEqual(lateral_of(2.0, "far"), "right")

    def test_walls(self):
        self.assertEqual(wall_region((5, 0.2)), "far_back")
        self.assertEqual(wall_region((0.2, 10)), "side_mesh")
        self.assertEqual(wall_region((0.2, 1.0)), "side_glass")
        self.assertIsNone(wall_region((5, 10)))


class BallTest(unittest.TestCase):
    def test_spike_removed_and_gap_filled(self):
        b = np.full((20, 3), np.nan, np.float32)
        for f in range(20):
            b[f] = (100 + 10 * f, 300, 1)
        b[7] = (900, 50, 1)        # spike
        b[12:14] = np.nan          # short gap
        out = clean_ball(b)
        # the spike is rejected and the 1-frame hole re-interpolated on the true path
        self.assertAlmostEqual(out[7, 0], 170, delta=1)
        self.assertAlmostEqual(out[7, 1], 300, delta=1)
        self.assertTrue(np.isfinite(out[12, 0]))
        self.assertAlmostEqual(out[12, 0], 220, delta=1)


class FeatureTest(unittest.TestCase):
    def test_mirror_symmetry(self):
        kp = np.zeros((31, 17, 3), np.float32)
        kp[..., 2] = 1
        kp[:, :, 0] = 500 + np.arange(17) * 3
        kp[:, :, 1] = 400 + np.arange(17) * 5
        ball = np.tile(np.array([[560, 380, 1]], np.float32), (31, 1))
        near = sequence_features(kp, ball, far=False, left_handed=False)
        self.assertEqual(near.shape, (31, N_FEAT))
        far = sequence_features(kp, ball, far=True, left_handed=False)
        # x coordinates are mirrored for far players, y untouched
        self.assertTrue(np.allclose(near[:, 0:51:3], -far[:, 0:51:3]))
        self.assertTrue(np.allclose(near[:, 1:51:3], far[:, 1:51:3]))


class ScoreboardTest(unittest.TestCase):
    def test_parse_and_changes(self):
        toks = [("JOS / SAN", 10, 50), ("15", 10, 200), ("SAL / TRI", 40, 50), ("0", 40, 200)]
        r = parse_ocr_tokens(toks)
        self.assertEqual(r["rows"][0]["points"], "15")
        # a reading must repeat 3 times (every 15 frames) before it is trusted
        states = [("0", "0"), ("15", "0"), ("15", "15")]
        samples, f = [], 0
        for a, b in states:
            for _ in range(3):
                samples.append({"frame": f, "rows": [{"games": [0], "points": a}, {"games": [0], "points": b}]})
                f += 15
        tl = {"samples": samples}
        ch = point_changes(tl)
        self.assertEqual([c["row"] for c in ch], [0, 1])


class ScoreboardLayoutTest(unittest.TestCase):
    def test_split_row_layouts(self):
        from padelcoach.scoreboard import split_row
        self.assertEqual(split_row({"nums": ["15"]}), ([], "15"))            # points-only box
        self.assertEqual(split_row({"nums": ["4"]}), ([4], None))           # games-only view
        self.assertEqual(split_row({"nums": ["6", "3", "40"]}), ([6, 3], "40"))
        self.assertEqual(split_row({"nums": ["6", "0"]}), ([6], "0"))
        self.assertEqual(split_row({"nums": ["0"]}), ([], None))            # ambiguous lone zero
        self.assertEqual(split_row({"games": [], "points": "4"}), ([4], None))  # old timeline format

    def test_resync_to_broadcast(self):
        from padelcoach.rally import _resync
        s = PadelScorer()
        s.award(0)
        _resync(s, [[6, 3], [1, 0]])
        self.assertEqual(s.state.sets, [[6, 3], [1, 0]])
        self.assertEqual(s.state.sets_won, [1, 0])
        self.assertEqual(s.state.points, [0, 0])
        _resync(s, [[6, 4]])                     # completed set -> new set opened
        self.assertEqual(s.state.sets[-1], [0, 0])


class BallLinkTest(unittest.TestCase):
    def test_trajectory_beats_isolated_false_positive(self):
        from padelcoach.perception import link_detector_candidates
        cands = {}
        for f in range(20):
            true = [100 + 15 * f, 300 + 2 * f, 0.35]          # consistent but low confidence
            cands[f] = np.array([true, [900, 800, 0.6 if f == 10 else 0.0]], np.float32)
        ball = link_detector_candidates(cands, 20)
        self.assertAlmostEqual(ball[10, 0], 250, delta=1)    # the trajectory wins at frame 10
        self.assertTrue(np.isfinite(ball[:, 0]).all())


if __name__ == "__main__":
    unittest.main()

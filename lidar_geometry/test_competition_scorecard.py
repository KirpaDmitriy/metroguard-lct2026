import unittest

from lidar_geometry.competition_scorecard import choose_default, count_episodes


class CompetitionScorecardTest(unittest.TestCase):
    def test_episode_boundaries(self):
        self.assertEqual(count_episodes([]), 0)
        self.assertEqual(count_episodes([False, True, True, False, True]), 2)

    def test_false_alarm_priority_is_lexicographic(self):
        report = {
            "normal_bags": {"models": {}},
            "official_pseudo_labels": {"models": {}},
            "composite_unseen_shapes": {"models": {}},
            "runtime_and_layouts": {"models": {}},
        }
        for name in ("geometry", "g4_linear", "portable_tree_25_75", "ood_guarded_g4"):
            report["normal_bags"]["models"][name] = {
                "alarm_frames": 1,
                "alarm_rate": 0.0005,
            }
            report["official_pseudo_labels"]["models"][name] = {
                "positive_coverage": 1.0
            }
            report["composite_unseen_shapes"]["models"][name] = {
                "mean_positive_recall": 1.0
            }
            report["runtime_and_layouts"]["models"][name] = {"p95_ms": 1.0}
        report["normal_bags"]["models"]["ood_guarded_g4"] = {
            "alarm_frames": 0,
            "alarm_rate": 0.0,
        }
        report["official_pseudo_labels"]["models"]["ood_guarded_g4"] = {
            "positive_coverage": 0.1
        }
        selected, _ = choose_default(report)
        self.assertEqual(selected, "ood_guarded_g4")

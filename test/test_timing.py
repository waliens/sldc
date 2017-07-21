from unittest import TestCase

import time

from sldc import WorkflowTiming, report_timing, StandardOutputLogger, Logger, merge_timings


class TestTiming(TestCase):
    def test_get_phase_hierarchy(self):
        phases = ["c.d", "a.b", "c", "c.d.e", "a.k"]
        splitted = [p.split(".") for p in phases]
        self.assertEqual(
            WorkflowTiming._get_phases_hierarchy(splitted),
            {"a": {"b": None, "k": None}, "c": {"d": {"e": None}}}
        )

    def testHierarchy(self):
        timing = WorkflowTiming()
        phase1, phase2, phase3 = "phase1", "phase2", "phase1.phase3"
        # Phases:
        # 1:    s ------- e  s -------- e
        # 2: s ------------------ e
        # 3:      s --- e      s --- e
        timing.start(phase2)
        timing.start(phase1)
        timing.start(phase3)
        timing.end(phase3)
        timing.end(phase1)
        timing.start(phase1)
        timing.start(phase3)
        timing.end(phase2)
        timing.end(phase3)
        timing.end(phase1)

        self.assertEqual(timing.get_phases_hierarchy(), {"phase1": {"phase3": None}, "phase2": None})
        self.assertEqual(2, len(timing.get(phase1)))
        self.assertEqual(1, len(timing.get(phase2)))
        self.assertEqual(2, len(timing.get(phase3)))

        # times are positive or null
        self.assertGreaterEqual(timing.get(phase1)[0], 0)
        self.assertGreaterEqual(timing.get(phase1)[1], 0)
        self.assertGreaterEqual(timing.get(phase2)[0], 0)
        self.assertGreaterEqual(timing.get(phase3)[0], 0)
        self.assertGreaterEqual(timing.get(phase3)[1], 0)

    def testTimingWithRoot(self):
        timing = WorkflowTiming(root="root")
        phase1, phase2, phase3 = "phase1", "phase2", "phase1.phase3"
        root_phase1, root_phase2, root_phase3 = "root.phase1", "root.phase2", "root.phase1.phase3"
        # Phases:
        # 1:    s ------- e  s -------- e
        # 2: s ------------------ e
        # 3:      s --- e      s --- e
        timing.start(phase2)
        timing.start(phase1)
        timing.start(phase3)
        timing.end(phase3)
        timing.end(phase1)
        timing.start(phase1)
        timing.start(phase3)
        timing.end(phase2)
        timing.end(phase3)
        timing.end(phase1)

        self.assertEqual(timing.get_phases_hierarchy(), {"root": {"phase1": {"phase3": None}, "phase2": None}})
        self.assertEqual(2, len(timing.get(root_phase1)))
        self.assertEqual(1, len(timing.get(root_phase2)))
        self.assertEqual(2, len(timing.get(root_phase3)))

    def testInvalidPhases(self):
        with self.assertRaises(ValueError):
            WorkflowTiming(root=".root")
        with self.assertRaises(ValueError):
            WorkflowTiming(root="root.")
        with self.assertRaises(ValueError):
            WorkflowTiming(root="r..oot")

        timing = WorkflowTiming()
        with self.assertRaises(ValueError):
            timing.start(".root")
        with self.assertRaises(ValueError):
            timing.start("root..")
        with self.assertRaises(ValueError):
            timing.start("ro...ot")

        with self.assertRaises(KeyError):
            timing.end("root")
        with self.assertRaises(KeyError):
            timing.total("root")
        with self.assertRaises(KeyError):
            timing.get_phase_statistics("root")

    def testMerge(self):
        timing1 = WorkflowTiming()
        timing2 = WorkflowTiming()

        timing1.start("root1")
        timing1.end("root1")
        timing1.start("root")
        timing1.end("root")
        timing1.start("root1.adj")
        timing1.end("root1.adj")
        timing2.start("root2")
        timing2.end("root2")
        timing2.start("root")
        timing2.end("root")
        timing2.start("root1.adj")
        timing2.end("root1.adj")

        # merge without update of initial objects
        merged = merge_timings(timing1, timing2)

        self.assertEqual(len(merged["root1"]), 1)
        self.assertEqual(len(merged["root2"]), 1)
        self.assertEqual(len(merged["root"]), 2)
        self.assertEqual(len(merged["root1.adj"]), 2)

        # merge with update of timing1
        timing1.merge(timing2)

        self.assertEqual(len(timing1["root1"]), 1)
        self.assertEqual(len(timing1["root2"]), 1)
        self.assertEqual(len(timing1["root"]), 2)
        self.assertEqual(len(timing1["root1.adj"]), 2)

        with self.assertRaises(TypeError):
            timing1.merge(StandardOutputLogger(level=Logger.DEBUG))

        timing1.merge(timing1)
        self.assertEqual(len(timing1["root1"]), 1)
        self.assertEqual(len(timing1["root2"]), 1)
        self.assertEqual(len(timing1["root"]), 2)
        self.assertEqual(len(timing1["root1.adj"]), 2)
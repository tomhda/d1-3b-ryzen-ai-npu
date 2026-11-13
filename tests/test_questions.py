"""Answer formatting and question validation (fake probabilities, no models)."""
import unittest

from d1npu.questions import format_answer, validate_questions, validate_state


class FormatNoul(unittest.TestCase):
    def test_basic(self):
        ans = format_answer("noul", ["yes", "no"], [0.8, 0.2])
        self.assertEqual(ans["type"], "noul")
        self.assertAlmostEqual(ans["noul"], 0.8)
        self.assertTrue(ans["decision"])

    def test_threshold(self):
        ans = format_answer("noul", ["yes", "no"], [0.8, 0.2], threshold=0.9)
        self.assertFalse(ans["decision"])
        ans = format_answer("noul", ["yes", "no"], [0.9, 0.1], threshold=0.9)
        self.assertTrue(ans["decision"])

    def test_invert(self):
        ans = format_answer("noul", ["yes", "no"], [0.8, 0.2], invert=True)
        self.assertAlmostEqual(ans["noul"], 0.2)
        self.assertFalse(ans["decision"])
        ans = format_answer(
            "noul", ["yes", "no"], [0.8, 0.2], threshold=0.75, invert=True)
        self.assertFalse(ans["decision"])
        ans = format_answer(
            "noul", ["yes", "no"], [0.1, 0.9], threshold=0.75, invert=True)
        self.assertAlmostEqual(ans["noul"], 0.9)
        self.assertTrue(ans["decision"])


class FormatChoice(unittest.TestCase):
    def test_pick_and_probabilities(self):
        ans = format_answer("choice", ["a", "b", "c"], [0.1, 0.7, 0.2])
        self.assertEqual(ans["type"], "choice")
        self.assertEqual(ans["choice"], "b")
        self.assertEqual(
            ans["probabilities"], {"a": 0.1, "b": 0.7, "c": 0.2})
        self.assertAlmostEqual(ans["confidence"], 0.7)


class FormatScore(unittest.TestCase):
    def test_expected_level(self):
        ans = format_answer("score", ["0", "1", "2"], [0.2, 0.3, 0.5])
        self.assertEqual(ans["type"], "score")
        self.assertAlmostEqual(ans["score"], 0.0 * 0.2 + 1.0 * 0.3 + 2.0 * 0.5)
        self.assertEqual(
            ans["probabilities"], {"0": 0.2, "1": 0.3, "2": 0.5})
        self.assertAlmostEqual(ans["confidence"], 0.5)
        self.assertEqual(ans["legend"], {"0": "0", "1": "1", "2": "2"})

    def test_legend_names_levels(self):
        ans = format_answer("score", ["0", "1"], [0.4, 0.6], legend=["low", "high"])
        self.assertEqual(ans["legend"], {"0": "low", "1": "high"})


class Validation(unittest.TestCase):
    def test_valid_mixed(self):
        out = validate_questions({
            "a": {"type": "noul", "instructions": "Is it?"},
            "b": {"type": "choice", "instructions": "Which?",
                  "criteria": {"x": "ex", "y": None}},
            "c": {"type": "score", "instructions": "How much?",
                  "criteria": ["low", "high"]},
        })
        self.assertEqual(out["a"]["threshold"], 0.5)
        self.assertEqual(out["a"]["invert"], False)

    def test_noul_options(self):
        out = validate_questions({
            "a": {"type": "noul", "instructions": "Is it absent?",
                  "threshold": 0.75, "invert": True},
        })
        self.assertEqual(out["a"]["threshold"], 0.75)
        self.assertTrue(out["a"]["invert"])

    def test_errors(self):
        bad = [
            ({}, "empty"),
            ({"a": {"instructions": "x"}}, "missing type"),
            ({"a": {"type": "maybe", "instructions": "x"}}, "bad type"),
            ({"a": {"type": "noul"}}, "missing instructions"),
            ({"a": {"type": "noul", "instructions": "  "}}, "blank"),
            ({"a": {"type": "choice", "instructions": "x",
                    "criteria": {"only": "one"}}}, "one option"),
            ({"a": {"type": "choice", "instructions": "x"}}, "no criteria"),
            ({"a": {"type": "score", "instructions": "x",
                    "criteria": ["only"]}}, "one level"),
            ({"a": {"type": "score", "instructions": "x",
                    "criteria": [f"l{i}" for i in range(11)]}}, "11 levels"),
            ({"a": {"type": "choice", "instructions": "x",
                    "criteria": {"p": "P", "q": "Q"}, "threshold": 0.5}},
             "threshold on choice"),
            ({"a": {"type": "score", "instructions": "x",
                    "criteria": ["l", "h"], "invert": True}}, "invert on score"),
            ({"a": {"type": "noul", "instructions": "x", "threshold": 2.0}},
             "threshold range"),
            ({"a": {"type": "noul", "instructions": "x", "invert": "yes"}},
             "invert type"),
            ({"a": {"type": "noul", "instructions": "x", "bogus": 1}},
             "unknown field"),
            ({"a": "not-a-dict"}, "question not a dict"),
        ]
        for questions, label in bad:
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    validate_questions(questions)


class StateValidation(unittest.TestCase):
    def test_str_and_list(self):
        self.assertEqual(validate_state("hello"), "hello")
        self.assertEqual(validate_state(["a", "b"]), ["a", "b"])

    def test_errors(self):
        for bad in (None, 42, {"text": "x"}, ["ok", 42], [["nested"]]):
            with self.subTest(repr(bad)):
                with self.assertRaises(ValueError):
                    validate_state(bad)


if __name__ == "__main__":
    unittest.main()

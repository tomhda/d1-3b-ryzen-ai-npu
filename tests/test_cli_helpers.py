"""CLI helpers and the bundled screen_check preset (no models)."""
import json
import tempfile
import unittest
from pathlib import Path

from d1npu.cli import (
    SCREEN_CHECK,
    _load_questions_file,
    _read_state,
    _screen_check_questions,
)
from d1npu.questions import validate_questions


class CliHelpers(unittest.TestCase):
    def test_read_state_plain(self):
        self.assertEqual(_read_state("hello"), "hello")

    def test_read_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.txt"
            path.write_text("from file", encoding="utf-8")
            self.assertEqual(_read_state(f"@{path}"), "from file")

    def test_load_questions_plain_and_wrapped(self):
        plain = {"a": {"type": "noul", "instructions": "x"}}
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "q1.json"
            p1.write_text(json.dumps(plain), encoding="utf-8")
            self.assertEqual(_load_questions_file(str(p1)), plain)
            p2 = Path(tmp) / "q2.json"
            p2.write_text(json.dumps({"questions": plain}), encoding="utf-8")
            self.assertEqual(_load_questions_file(str(p2)), plain)

    def test_screen_check_preset_is_valid(self):
        self.assertTrue(SCREEN_CHECK.exists())
        state, questions, order = _screen_check_questions()
        self.assertTrue(state)
        self.assertEqual(len(order), len(questions))
        for key in ("login_form", "page", "error", "modal", "menu"):
            self.assertIn(key, questions)
        valid = validate_questions(questions)
        self.assertEqual(set(valid), set(order))
        modal = valid["modal"]
        self.assertTrue(modal["invert"])
        self.assertEqual(modal["threshold"], 0.75)


if __name__ == "__main__":
    unittest.main()

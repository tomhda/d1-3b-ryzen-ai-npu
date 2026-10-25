"""Tree-evaluation guards with a fake engine: max_seq_len and cache checks."""
import copy
import io
import sys
import types
import unittest

from d1npu.engine import digest_cache
from d1npu.questions import check_seq_len


class FakeEngine:
    """Mimics D1._evaluate_tree: trunk once, each branch on a private copy."""

    def __init__(self, max_seq_len=4096, mutate=False):
        self.max_seq_len = max_seq_len
        self.mutate = mutate

    def forward(self, ids, past=None, past_len=0, cache=False):
        check_seq_len(past_len, len(ids), self.max_seq_len)
        if past is None:
            state = {"past_key_values.0.key": bytearray(b"trunk-state")}
        else:
            state = {k: bytearray(v) for k, v in past.items()}
            if self.mutate:
                past["past_key_values.0.key"][:] = b"corrupted!!"
        return [0.7, 0.3], copy.deepcopy(state) if cache else None

    def evaluate_tree(self, trunk, branches):
        _, past = self.forward(trunk, cache=True)
        before = digest_cache(past)
        out = {}
        for key, branch in branches.items():
            probs, _ = self.forward(branch, past, len(trunk))
            out[key] = probs
        after = digest_cache(past)
        assert before == after, "shared cache was mutated"
        return out


class SeqLenGuard(unittest.TestCase):
    def test_ok_at_limit(self):
        check_seq_len(100, 3996, 4096)

    def test_exceeds(self):
        with self.assertRaises(ValueError):
            check_seq_len(100, 3997, 4096)

    def test_fake_engine_enforces(self):
        engine = FakeEngine(max_seq_len=10)
        with self.assertRaises(ValueError):
            engine.evaluate_tree([0] * 8, {"q": [1] * 3})


class CacheCheck(unittest.TestCase):
    def test_digest_stable(self):
        past = {"b": b"xyz", "a": b"abc"}
        self.assertEqual(digest_cache(past), digest_cache(dict(past)))

    def test_digest_detects_mutation(self):
        past = {"a": bytearray(b"abc")}
        before = digest_cache(past)
        past["a"][:] = b"abd"
        self.assertNotEqual(before, digest_cache(past))

    def test_tree_keeps_cache(self):
        engine = FakeEngine()
        out = engine.evaluate_tree([0] * 5, {"q1": [1], "q2": [2, 3]})
        self.assertEqual(set(out), {"q1", "q2"})

    def test_tree_detects_mutation(self):
        engine = FakeEngine(mutate=True)
        with self.assertRaises(AssertionError):
            engine.evaluate_tree([0] * 5, {"q1": [1]})


class ServeResponse(unittest.TestCase):
    def test_success_shape(self):
        from d1npu.server import build_response

        def decide(state, questions, image):
            self.assertEqual(state, "hello")
            self.assertIsNone(image)
            answers = {
                "a": {"type": "noul", "noul": 0.8, "decision": True}}
            timing = {"vision_s": 0.0, "trunk_s": 0.1, "branches_s": 0.2,
                      "total_s": 0.3, "trunk_tokens": 9, "input_tokens": 15}
            return answers, timing

        status, payload = build_response(
            decide, {"model": "x", "state": "hello",
                     "questions": {"a": {"type": "noul",
                                        "instructions": "Is it?"}}})
        self.assertEqual(status, 200)
        self.assertEqual(payload["model"], "d1-3b-ryzen-ai-npu")
        self.assertEqual(payload["answers"]["a"]["decision"], True)
        self.assertEqual(payload["usage"],
                         {"input_tokens": 15, "output_tokens": 0})
        self.assertEqual(payload["timing"]["trunk_tokens"], 9)

    def test_bad_request_is_400(self):
        from d1npu.server import build_response

        def decide(state, questions, image):  # pragma: no cover
            raise AssertionError("must not run")

        for body in ({"state": "s", "questions": {}},
                     {"state": "s", "questions": {"a": {"type": "bogus"}}},
                     {"state": ["a", "b", 1], "questions": {
                         "a": {"type": "noul", "instructions": "x"}}}):
            status, payload = build_response(decide, body)
            self.assertEqual(status, 400)
            self.assertIn("error", payload)

    def test_image_reaches_decide(self):
        from d1npu.server import build_response

        import base64
        from d1npu.request_parsing import decode_data_url  # noqa: F401

        raw = base64.b64decode("aGk=")  # b"hi"
        url = "data:image/png;base64," + base64.b64encode(raw).decode()
        seen = {}

        def decide(state, questions, image):
            seen["image"] = image
            return {}, {"input_tokens": 1}

        sentinel = object()

        class FakeImage:
            @staticmethod
            def open(buf):
                assert isinstance(buf, io.BytesIO)
                assert buf.getvalue() == raw

                class Img:
                    def convert(self, mode):
                        assert mode == "RGB"
                        return sentinel

                return Img()

        pil = types.ModuleType("PIL")
        pil.Image = FakeImage
        old = sys.modules.get("PIL")
        sys.modules["PIL"] = pil
        try:
            status, _ = build_response(
                decide,
                {"state": ["look", {"type": "image_url",
                                   "image_url": {"url": url}}],
                 "questions": {"a": {"type": "noul", "instructions": "x"}}})
        finally:
            if old is None:
                del sys.modules["PIL"]
            else:
                sys.modules["PIL"] = old
        self.assertEqual(status, 200)
        self.assertIs(seen["image"], sentinel)

    def test_undecodable_image_is_400(self):
        from d1npu.server import build_response

        import base64

        url = "data:image/png;base64," + base64.b64encode(b"hi").decode()

        class FakeImage:
            @staticmethod
            def open(buf):
                raise OSError("cannot identify image file")

        pil = types.ModuleType("PIL")
        pil.Image = FakeImage
        old = sys.modules.get("PIL")
        sys.modules["PIL"] = pil
        try:
            status, body = build_response(
                lambda *a: self.fail("decide must not run"),
                {"state": ["look", {"type": "image_url",
                                   "image_url": {"url": url}}],
                 "questions": {"a": {"type": "noul", "instructions": "x"}}})
        finally:
            if old is None:
                del sys.modules["PIL"]
            else:
                sys.modules["PIL"] = old
        self.assertEqual(status, 400)
        self.assertIn("image could not be decoded", body["error"])


if __name__ == "__main__":
    unittest.main()

"""Server request parsing: data URLs, image limit, string items."""
import base64
import unittest

from d1npu.request_parsing import decode_data_url, parse_body, parse_state

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
JPG_URL = "data:image/jpeg;base64," + base64.b64encode(PNG_BYTES).decode()


def image_part(url=PNG_URL):
    return {"type": "image_url", "image_url": {"url": url}}


class DataUrl(unittest.TestCase):
    def test_png(self):
        self.assertEqual(decode_data_url(PNG_URL), PNG_BYTES)

    def test_jpeg(self):
        self.assertEqual(decode_data_url(JPG_URL), PNG_BYTES)

    def test_errors(self):
        for bad in ("http://x/y.png", "data:text/plain;base64,xx",
                    "data:image/png;base64,", "data:image/png;base64,!!!",
                    "data:image/gif;base64,xx", None, 42):
            with self.subTest(repr(bad)[:40]):
                with self.assertRaises(ValueError):
                    decode_data_url(bad)


class ParseState(unittest.TestCase):
    def test_plain_string(self):
        text, image = parse_state("hello")
        self.assertEqual(text, "hello")
        self.assertIsNone(image)

    def test_single_text_item(self):
        text, image = parse_state(["hello"])
        self.assertEqual(text, "hello")
        self.assertIsNone(image)

    def test_texts_and_one_image(self):
        text, image = parse_state(["a", image_part(), "b"])
        self.assertEqual(text, ["a", "b"])
        self.assertEqual(image, PNG_BYTES)

    def test_image_only(self):
        text, image = parse_state([image_part()])
        self.assertEqual(text, "")
        self.assertEqual(image, PNG_BYTES)

    def test_two_images_rejected(self):
        with self.assertRaises(ValueError):
            parse_state([image_part(), image_part()])

    def test_non_string_items_rejected(self):
        with self.assertRaises(ValueError):
            parse_state(["ok", 42])
        with self.assertRaises(ValueError):
            parse_state([{"type": "text", "text": "x"}])
        with self.assertRaises(ValueError):
            parse_state([{"type": "image_url"}])
        with self.assertRaises(ValueError):
            parse_state([{"type": "image_url",
                          "image_url": {"url": "http://x"}}])

    def test_wrong_type_rejected(self):
        for bad in (None, 42, {"text": "x"}):
            with self.subTest(repr(bad)):
                with self.assertRaises(ValueError):
                    parse_state(bad)


class ParseBody(unittest.TestCase):
    def test_ok(self):
        state, questions = parse_body({"model": "m", "state": "s",
                                       "questions": {"q": {}}})
        self.assertEqual(state, "s")
        self.assertEqual(questions, {"q": {}})

    def test_errors(self):
        for bad in ([], "x", {}, {"state": "s"},
                    {"questions": {}}, {"state": "s", "questions": {}}):
            with self.subTest(repr(bad)[:40]):
                with self.assertRaises(ValueError):
                    parse_body(bad)


if __name__ == "__main__":
    unittest.main()

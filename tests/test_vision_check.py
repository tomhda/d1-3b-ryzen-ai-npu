"""Vision grid checks with fake processor outputs (no models, no PIL)."""
import os
import tempfile
import unittest

from d1npu.vision_check import (
    VISION_VALID_PATCHES,
    validate_vision_grid,
    vision_cache_key,
)


class FakeTensor:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value

    def sum(self, axis=None):
        total = self._value
        if isinstance(total, list):
            flat = []
            stack = [total]
            while stack:
                item = stack.pop()
                if isinstance(item, list):
                    stack.extend(item)
                else:
                    flat.append(item)
            return sum(flat)
        return total


class VisionGrid(unittest.TestCase):
    def test_valid_lists(self):
        validate_vision_grid([[24, 40]], [[1] * 960])

    def test_valid_count_int(self):
        validate_vision_grid([[24, 40]], 960)

    def test_valid_fake_tensors(self):
        spatial = FakeTensor([[24, 40]])
        mask = FakeTensor([[1] * 960])
        validate_vision_grid(spatial, mask)

    def test_wrong_spatial(self):
        with self.assertRaises(ValueError):
            validate_vision_grid([[20, 20]], 960)

    def test_wrong_count(self):
        with self.assertRaises(ValueError):
            validate_vision_grid([[24, 40]], 959)
        with self.assertRaises(ValueError):
            validate_vision_grid([[24, 40]], [[1] * 959 + [0]])

    def test_valid_constant(self):
        self.assertEqual(VISION_VALID_PATCHES, 960)


class CacheKey(unittest.TestCase):
    def test_changes_with_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vision_encoder_npu.onnx")
            with open(path, "wb") as f:
                f.write(b"x" * 100)
            first = vision_cache_key(path)
            self.assertIn("100", first)
            with open(path, "ab") as f:
                f.write(b"y")
            os.utime(path, (1_700_000_000, 1_700_000_100))
            second = vision_cache_key(path)
            self.assertNotEqual(first, second)
            self.assertIn("101", second)


if __name__ == "__main__":
    unittest.main()

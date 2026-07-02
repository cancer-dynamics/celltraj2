import unittest

from celltraj2.paths import frame_key, label_frame_path, mask_frame_path, parse_frame_key, validate_name


class PathTests(unittest.TestCase):
    def test_frame_keys_are_one_based(self):
        self.assertEqual(frame_key(1), "frame_1")
        self.assertEqual(frame_key(12), "frame_12")
        self.assertEqual(parse_frame_key("frame_12"), 12)
        with self.assertRaises(ValueError):
            frame_key(0)

    def test_named_sets_are_single_path_segments(self):
        self.assertEqual(validate_name("epithelial"), "epithelial")
        self.assertEqual(validate_name("immune.v1"), "immune.v1")
        with self.assertRaises(ValueError):
            validate_name("0bad")
        with self.assertRaises(ValueError):
            validate_name("bad/name")

    def test_frame_paths(self):
        self.assertEqual(label_frame_path("epithelial", 1), "/labels/epithelial/frame_1")
        self.assertEqual(mask_frame_path("nuclear", 2), "/masks/nuclear/frame_2")


if __name__ == "__main__":
    unittest.main()

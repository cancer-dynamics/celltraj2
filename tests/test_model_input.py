import unittest

from celltraj2.model_input import compose_model_input


class ModelInputTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        self.np = np

    def test_compose_3d_model_input_returns_zcyx_for_multiple_outputs(self):
        data = self.np.zeros((2, 3, 4, 3), dtype=self.np.float32)
        data[..., 0] = 2.0
        data[..., 1] = 5.0
        data[..., 2] = self.np.arange(2 * 3 * 4, dtype=self.np.float32).reshape(2, 3, 4)

        output = compose_model_input(
            data,
            axes=("Z", "Y", "X", "C"),
            do_3d=True,
            channel_specs=[
                {"channel_indices": [0], "normalization": "raw", "combination": "single"},
                {"channel_indices": [1, 2], "normalization": "full_uint16", "combination": "max"},
            ],
        )

        self.assertEqual(output.shape, (2, 2, 3, 4))
        self.assertTrue(self.np.all(output[:, 0] == 2.0))
        self.assertEqual(int(output[:, 1].max()), 65535)

    def test_lut_scaling_uses_source_channel_metadata(self):
        data = self.np.arange(6, dtype=self.np.float32).reshape(1, 2, 3, 1)

        output = compose_model_input(
            data,
            axes=("Z", "Y", "X", "C"),
            do_3d=True,
            channel_specs=[
                {
                    "channel_indices": [0],
                    "normalization": "lut_full_uint16",
                    "source_channels": [
                        {"raw_index": 0, "lut": {"low_cutoff": 1.0, "high_cutoff": 5.0}},
                    ],
                }
            ],
        )

        self.assertEqual(output.shape, (1, 2, 3))
        self.assertEqual(int(output[0, 0, 0]), 0)
        self.assertEqual(int(output[0, 1, 2]), 65535)

    def test_2d_model_input_requires_z_index_for_stack(self):
        data = self.np.zeros((2, 3, 4, 1), dtype=self.np.uint16)

        with self.assertRaises(ValueError):
            compose_model_input(
                data,
                axes=("Z", "Y", "X", "C"),
                do_3d=False,
                channel_specs=[{"channel_indices": [0], "normalization": "raw"}],
            )

        output = compose_model_input(
            data,
            axes=("Z", "Y", "X", "C"),
            do_3d=False,
            z_index=1,
            channel_specs=[{"channel_indices": [0], "normalization": "raw"}],
        )
        self.assertEqual(output.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()

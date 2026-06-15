from __future__ import annotations

import unittest

from api.geometry import (
    angle_cw_deg,
    needle_tip_from_mask,
    normalized_reading_from_angles,
)

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


class GeometryTest(unittest.TestCase):
    def test_angle_cw_uses_image_coordinates(self) -> None:
        center = (10.0, 10.0)
        self.assertAlmostEqual(angle_cw_deg(center, (10.0, 0.0)), 0.0)
        self.assertAlmostEqual(angle_cw_deg(center, (20.0, 10.0)), 90.0)
        self.assertAlmostEqual(angle_cw_deg(center, (10.0, 20.0)), 180.0)

    def test_normalized_reading_across_zero_angle(self) -> None:
        reading = normalized_reading_from_angles(
            start_angle=300.0,
            end_angle=60.0,
            needle_angle=0.0,
        )
        self.assertAlmostEqual(reading, 0.5)

    def test_normalized_reading_clamps_to_nearest_endpoint(self) -> None:
        self.assertEqual(
            normalized_reading_from_angles(
                start_angle=300.0,
                end_angle=60.0,
                needle_angle=250.0,
            ),
            0.0,
        )
        self.assertEqual(
            normalized_reading_from_angles(
                start_angle=300.0,
                end_angle=60.0,
                needle_angle=100.0,
            ),
            1.0,
        )

    @unittest.skipIf(np is None, "numpy is not installed in the no-sync test env")
    def test_needle_tip_uses_farthest_mask_pixels(self) -> None:
        assert np is not None
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10:19] = True
        tip = needle_tip_from_mask(mask, center=(10.0, 10.0))
        self.assertEqual(tip, (18.0, 10.0))


if __name__ == "__main__":
    unittest.main()

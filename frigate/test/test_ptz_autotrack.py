import unittest

import numpy as np

from frigate.ptz.autotrack import transform_is_finite


class DummyTransform:
    pass


class TestPtzAutotrack(unittest.TestCase):
    def test_transform_is_finite_all_none(self):
        coord_transformations = DummyTransform()
        self.assertTrue(transform_is_finite(coord_transformations))

    def test_transform_is_finite_finite_values(self):
        coord_transformations = DummyTransform()
        coord_transformations.homography_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        coord_transformations.inverse_homography_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        coord_transformations.movement_vector = np.array([0.0, 0.0])
        self.assertTrue(transform_is_finite(coord_transformations))

    def test_transform_is_finite_nan(self):
        coord_transformations = DummyTransform()
        coord_transformations.homography_matrix = np.array(
            [[1.0, np.nan, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        coord_transformations.inverse_homography_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        coord_transformations.movement_vector = np.array([0.0, 0.0])
        self.assertFalse(transform_is_finite(coord_transformations))

    def test_transform_is_finite_inf(self):
        coord_transformations = DummyTransform()
        coord_transformations.homography_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        coord_transformations.inverse_homography_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, 1.0]]
        )
        coord_transformations.movement_vector = np.array([0.0, 0.0])
        self.assertFalse(transform_is_finite(coord_transformations))

    def test_transform_is_finite_neginf(self):
        coord_transformations = DummyTransform()
        coord_transformations.homography_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        coord_transformations.inverse_homography_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        coord_transformations.movement_vector = np.array([-np.inf, 0.0])
        self.assertFalse(transform_is_finite(coord_transformations))

import numpy as np

from tools.analysis.analyze_search_coverage import visible_samples


def test_visibility_requires_front_projection_and_depth_beyond_sample():
    samples = np.array([[1.0, 0, 0], [3.0, 0, 0], [-1.0, 0, 0], [1.0, 10, 0]])
    visible = visible_samples(
        samples,
        np.full((12, 16), 2.0),
        np.zeros(3),
        np.array([1.0, 0, 0]),
        np.array([0.0, -1, 0]),
        np.array([0.0, 0, 1]),
        8.0,
    )
    assert visible.tolist() == [True, False, False, False]


def test_missing_depth_and_foreground_patch_do_not_certify_space():
    args = (
        np.array([[1.0, 0, 0]]),
        np.full((12, 16), np.nan),
        np.zeros(3),
        np.array([1.0, 0, 0]),
        np.array([0.0, -1, 0]),
        np.array([0.0, 0, 1]),
        8.0,
    )
    assert not visible_samples(*args)[0]
    args[1][5:8, 7:10] = 2.0
    args[1][5, 7:10] = 0.5
    assert not visible_samples(*args)[0]

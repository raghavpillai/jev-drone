"""Thin occupied pixels must constrain free-space sampling."""
import numpy as np

from jev_drone.perception.depth import clearance
from jev_drone.perception.sensors import Frame


def frame():
    return Frame(0., np.zeros((120, 160, 3)), np.full((120, 160), 5.),
                 np.zeros(3), np.array([1., 0., 0.]), np.array([0., -1., 0.]),
                 np.array([0., 0., 1.]), 32.)


def test_thin_off_center_obstacle_between_sampled_footprint_rays():
    image = frame()
    # A narrow vertical pole 0.65 m ahead and about 0.20 m to the side.
    image.depth[:, 70:72] = .65
    assert clearance([image], np.zeros(3), np.array([1., 0., 0.])) < .4


def test_free_view_remains_usable_and_inputs_are_not_normalized_in_place():
    direction = np.array([2., 0., 0.])
    assert clearance([frame()], np.zeros(3), direction) >= 1.3
    assert np.array_equal(direction, [2., 0., 0.])


def test_known_obstacle_caps_clearance_even_when_another_camera_sees_behind_it():
    close = frame()
    close.depth[:, 70:72] = .65
    assert clearance([frame(), close], np.zeros(3), np.array([1., 0., 0.])) < .4


def test_missing_depth_does_not_become_free_and_motion_away_is_possible():
    assert clearance([], np.zeros(3), np.array([1., 0., 0.])) is None
    close = frame()
    close.depth[:] = np.nan
    assert clearance([close], np.zeros(3), np.array([1., 0., 0.])) is None
    front, back = frame(), frame()
    back.forward *= -1
    back.right *= -1
    back.depth[:] = .2
    assert clearance([front, back], np.zeros(3), np.array([1., 0., 0.])) >= 1.3

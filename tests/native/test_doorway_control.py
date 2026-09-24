from jev_drone.control.doorway_control import describe, heading


def test_known_door_crossing_heading_in_both_directions():
    assert heading({"name":"cross_living_bedroom", "position":[7.,2.,1.3]}) == 0.
    assert heading({"name":"cross_living_bedroom", "position":[5.,2.,1.3]}) == 180.
    assert heading({"name":"cross_bedroom_study", "position":[9.,5.,1.3]}) == 90.
    assert heading({"name":"cross_bedroom_study", "position":[9.,3.,1.3]}) == -90.
    assert heading({"name":"approach_living_bedroom", "position":[5.,2.,1.3]}) is None


def test_alignment_exposes_centerline_error_without_filtering_controls():
    state = {"task":{"name":"cross_living_bedroom", "position":[5.,2.,1.3]},
        "sensors":{"position_estimate":[6.8,2.4,1.6], "yaw_degrees":100.}}
    criteria = dict(pan_left="Turn", detour_short_right="Bypass", brake="Stop")
    enriched, described = describe(state, criteria)
    assert enriched["doorway_alignment"]["heading_error_degrees"] == 80.
    assert enriched["doorway_alignment"]["centerline_offset_m"] == .4
    assert set(described) == set(criteria)
    assert "doorway_alignment" not in state

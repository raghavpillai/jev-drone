from dataclasses import dataclass

from jev_drone.world.world import Config as PolicyConfig


@dataclass
class Config(PolicyConfig):
    find_only: bool = False
    disturbance_newtons: float = 0.
    action_facts: bool = False
    control_interface: str = "patch"
    front_controls: bool = True
    control_schema: str = "joystick-v1"
    pilot_representation: str = "intent-v36-local-progress"
    command_lease_seconds: float = .9
    slow_clearance_m: float = .7
    creep_clearance_m: float = .4
    full_clearance_m: float = 1.5
    stopping_margin: bool = True
    braking_settling_allowance_s: float = .7
    patch_encoding: str = "changed_channels"
    goal_horizontal_tolerance_m: float = .35
    goal_vertical_tolerance_m: float = .15

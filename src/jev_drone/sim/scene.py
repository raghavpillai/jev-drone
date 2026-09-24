"""Export benchmark geometry and a real X500 motor/sensor model to SDF."""
from copy import deepcopy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from jev_drone.world.world import Config, World, DIRECTIONS
from jev_drone.world.layout import for_case


WIDTH, HEIGHT, HFOV = 160, 120, 2.4
# Cameras are fixed to the vehicle, including its actual pitch and roll.
CAMERAS = {
    "east": ((.30, 0, .10), (0, 0, 0)),
    "west": ((-.30, 0, .10), (0, 0, math.pi)),
    "north": ((0, .30, .10), (0, 0, math.pi/2)),
    "south": ((0, -.30, .10), (0, 0, -math.pi/2)),
    "up": ((0, 0, .22), (0, -math.pi/2, 0)),
    "down": ((0, 0, -.26), (0, math.pi/2, 0)),
}


def camera_rig(config):
    return CAMERAS, HFOV


def element(parent, tag, text=None, **attributes):
    item = ET.SubElement(parent, tag, attributes)
    if text is not None:
        item.text = str(text)
    return item


def numbers(values):
    return " ".join(str(float(x)) for x in values)


class Geometry(World):
    """Reuse only the existing scene recipe; never start a PyBullet simulation."""
    def __init__(self, config):
        self.config = config
        self.layout = for_case(config.case)
        self.rng = np.random.default_rng(config.seed)
        self.geometry, self.dynamic, self.targets = [], [], {}
        self.start = np.array([1.4, 2., 1.3])
        self._build()
        if getattr(config, "find_only", False):
            self.mission = [objective for objective in self.mission if "marker" in objective][:1]
            if not self.mission:
                raise ValueError("Find-only missions require an item objective")

    def box(self, name, center, half, color=(.58, .57, .55, 1), dynamic=False):
        body = len(self.geometry)
        self.geometry.append(dict(name=name, center=list(center), half=list(half),
                                  color=list(color), body=body, dynamic=dynamic))
        return body


def generate(config, source, output):
    output.mkdir(parents=True, exist_ok=True)
    scene = Geometry(config)
    for actor in scene.dynamic:
        scene.geometry[actor["body"]]["center"][1] = 2+1.35*math.sin(actor["phase"])
    sdf = ET.Element("sdf", version="1.9")
    world = element(sdf, "world", name="jev")
    physics = element(world, "physics", name="physics", type="ode")
    element(physics, "max_step_size", .004)
    element(physics, "real_time_factor", 1)
    element(world, "gravity", "0 0 -9.81")
    element(world, "magnetic_field", "6e-6 2.3e-5 -4.2e-5")
    for library, name in (("physics", "Physics"), ("user-commands", "UserCommands"),
                          ("scene-broadcaster", "SceneBroadcaster"), ("contact", "Contact"),
                          ("imu", "Imu"), ("air-pressure", "AirPressure"),
                          ("magnetometer", "Magnetometer"), ("sensors", "Sensors"),
                          ("apply-link-wrench", "ApplyLinkWrench")):
        plugin = element(world, "plugin", filename=f"gz-sim-{library}-system", name=f"gz::sim::systems::{name}")
        if name == "Sensors":
            element(plugin, "render_engine", "ogre2")
    render = element(world, "scene")
    element(render, "ambient", ".7 .7 .7 1")
    element(render, "background", ".6 .7 .8 1")
    element(render, "shadows", "false")
    light = element(world, "light", name="sun", type="directional")
    element(light, "direction", ".2 .1 -1")
    element(light, "diffuse", ".8 .8 .8 1")
    element(light, "cast_shadows", "false")
    for i, box in enumerate(scene.geometry):
        model = element(world, "model", name=f"{box['name']}_{i}")
        element(model, "static", "true")
        element(model, "pose", numbers(box["center"]+[0, 0, 0]))
        link = element(model, "link", name="link")
        for kind in ("collision", "visual"):
            node = element(link, kind, name=kind)
            geometry = element(node, "geometry")
            element(element(geometry, "box"), "size", numbers(np.array(box["half"])*2))
            if kind == "visual":
                material = element(node, "material")
                for field in ("ambient", "diffuse"):
                    element(material, field, numbers(box["color"]))
    models = source/"Tools/simulation/gz/models"
    drone = deepcopy(ET.parse(models/"x500_base/model.sdf").getroot().find("model"))
    drone.set("name", "drone")
    drone.find("pose").text = numbers([*scene.start[:2], .24, 0, 0, 0])
    base = drone.find("link[@name='base_link']")
    for sensor in list(base.findall("sensor[@type='navsat']")):
        base.remove(sensor)
    for plugin in ET.parse(models/"x500/model.sdf").getroot().findall("model/plugin"):
        drone.append(deepcopy(plugin))
    odometry = element(drone, "plugin", filename="gz-sim-odometry-publisher-system",
                       name="gz::sim::systems::OdometryPublisher")
    element(odometry, "dimensions", 3)
    element(odometry, "odom_publish_frequency", 50)
    element(odometry, "gaussian_noise", .005)
    cameras, hfov = camera_rig(config)
    for name, (position, angles) in cameras.items():
        sensor = element(base, "sensor", name="camera_"+name, type="rgbd_camera")
        element(sensor, "pose", numbers(position+angles))
        element(sensor, "topic", "/jev/camera/"+name)
        element(sensor, "always_on", "true")
        element(sensor, "update_rate", config.sensor_hz)
        camera = element(sensor, "camera")
        element(camera, "horizontal_fov", hfov)
        size = element(camera, "image")
        element(size, "width", WIDTH)
        element(size, "height", HEIGHT)
        clip = element(camera, "clip")
        element(clip, "near", .08)
        element(clip, "far", 12)
    for link in drone.findall("link"):
        collisions = link.findall("collision")
        if collisions:
            sensor = element(link, "sensor", name="contacts", type="contact")
            element(sensor, "topic", "/jev/contacts/"+link.get("name"))
            element(sensor, "always_on", "true")
            contact = element(sensor, "contact")
            # Gazebo Harmonic's Contact system reads contact/topic, unlike
            # camera and IMU systems which read sensor/topic.
            element(contact, "topic", "/jev/contacts/"+link.get("name"))
            for collision in collisions:
                element(contact, "collision", collision.get("name"))
    world.append(drone)
    ET.indent(sdf)
    ET.ElementTree(sdf).write(output/"world.sdf", encoding="unicode", xml_declaration=True)
    metadata = {"geometry": scene.geometry, "mission": scene.mission, "targets": scene.targets,
                "start": scene.start.tolist(), "actors": scene.dynamic,
                "rooms": scene.layout.rooms, "doors": scene.layout.doors,
                "room_connections": scene.layout.connections, "cameras": cameras, "width": WIDTH, "height": HEIGHT,
                "hfov": hfov, "localization": "PX4 EKF2 fusing simulated external odometry; no GPS, no real VIO",
                "depth": "Gazebo RGB-D renderer, not learned monocular or stereo depth"}
    (output/"scene.json").write_text(json.dumps(metadata, indent=2))
    return metadata

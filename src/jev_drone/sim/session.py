"""Own the Gazebo and PX4 subprocesses; every trial gets a fresh simulator."""

import multiprocessing
import os
import queue
import signal
import subprocess
import time
import uuid
from pathlib import Path

from jev_drone.sim.actors import animate
from jev_drone.sim.flight import Flight
from jev_drone.sim.scene import generate
from jev_drone.sim.transport import Transport


class Session:
    def __init__(self, config, output, source=Path(".sim/PX4-Autopilot")):
        # Gazebo discovers peers across Docker's bridge; network namespaces alone
        # do not isolate equal world/topic names in concurrent experiments.
        os.environ["GZ_PARTITION"] = "jev_" + uuid.uuid4().hex
        self.output = output.resolve()
        self.source = source.resolve()
        self.processes = []
        self.streams = []
        self.flight = None
        self.config = config
        self.motion_process = None
        self._motion_error = None
        self.context = multiprocessing.get_context("spawn")
        self.motion_time = self.context.Value("d", 0.0, lock=False)
        self.motion_epoch = self.context.Value("d", 0.0, lock=False)
        self.motion_active = self.context.Value("b", False, lock=False)
        self.motion_running = self.context.Value("b", True, lock=False)
        self.motion_messages = self.context.Queue()
        self.scene = generate(config, self.source, self.output)
        self.transport = Transport(self.motion_time)

    def launch(self):
        models = self.source / "Tools/simulation/gz/models"
        env = dict(
            os.environ,
            GZ_SIM_RESOURCE_PATH=str(models),
            GZ_SIM_SYSTEM_PLUGIN_PATH=str(
                self.source / "build/px4_sitl_default/src/modules/simulation/gz_plugins"
            ),
        )
        self._start(
            [
                "gz",
                "sim",
                "-r",
                "-s",
                "--headless-rendering",
                "--force-version",
                "8",
                str(self.output / "world.sdf"),
            ],
            "gazebo",
            env,
            self.output,
        )
        deadline = time.monotonic() + 60
        while self.transport.time < 1:
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("Gazebo did not start; inspect gazebo.log")
            time.sleep(0.1)
        build = self.source / "build/px4_sitl_default"
        rootfs = self.output / "rootfs"
        rootfs.mkdir()
        env.update(
            PX4_SYS_AUTOSTART="4001",
            PX4_GZ_STANDALONE="1",
            PX4_GZ_WORLD="jev",
            PX4_GZ_MODEL_NAME="drone",
            PX4_PARAM_EKF2_GPS_CTRL="0",
            PX4_PARAM_EKF2_EV_CTRL="15",
            PX4_PARAM_EKF2_HGT_REF="3",
            PX4_PARAM_EKF2_EV_NOISE_MD="1",
            PX4_PARAM_EKF2_EVP_NOISE="0.03",
            PX4_PARAM_EKF2_EVV_NOISE="0.05",
            PX4_PARAM_EKF2_MAG_TYPE="5",
            PX4_PARAM_COM_ARM_WO_GPS="1",
            PX4_PARAM_COM_RC_IN_MODE="4",
            PX4_PARAM_NAV_DLL_ACT="0",
            PX4_PARAM_COM_OF_LOSS_T="0.5",
            PX4_PARAM_COM_OBL_RC_ACT="4",
            PX4_PARAM_COM_DISARM_LAND="2",
        )
        self._start([str(build / "bin/px4"), "-d", str(build / "etc")], "px4", env, rootfs)
        self.flight = Flight(
            lambda: self.transport.time,
            body_controls=getattr(self.config, "front_controls", False),
            lease=self.config.command_lease_seconds,
        )
        self.flight.connect()
        if self.scene["actors"] or self.config.case == "moved_target":
            self.motion_process = self.context.Process(
                target=animate,
                args=(
                    self.scene,
                    self.config.case,
                    self.motion_time,
                    self.motion_epoch,
                    self.motion_active,
                    self.motion_running,
                    self.motion_messages,
                ),
            )
            self.motion_process.start()
            error = self.motion_messages.get(timeout=12)
            if error:
                raise RuntimeError(error)
        return self

    def _start(self, command, name, env, cwd):
        stream = (self.output / (name + ".log")).open("w")
        self.streams.append(stream)
        self.processes.append(
            subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )

    def check(self):
        for process in self.processes:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Simulator process exited with {process.returncode}; inspect logs"
                )

    def start_mission(self, epoch):
        force = getattr(self.config, "disturbance_newtons", 0)
        if force:
            self.transport.apply_force(force)
        self.motion_epoch.value = epoch
        self.motion_active.value = True

    @property
    def motion_error(self):
        try:
            self._motion_error = self.motion_messages.get_nowait()
        except queue.Empty:
            pass
        return self._motion_error

    def close(self):
        self.motion_running.value = False
        if self.motion_process:
            self.motion_process.join(timeout=2)
            if self.motion_process.is_alive():
                self.motion_process.terminate()
                self.motion_process.join(timeout=2)
        if self.flight:
            self.flight.close()
        for process in reversed(self.processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        for stream in self.streams:
            stream.close()

"""Vehicle profiles. Sources are in the comments; fields marked (est.) are estimates with no published value."""
import math
from dataclasses import dataclass

import numpy as np

G = 9.80665
RHO = 1.225


@dataclass(frozen=True)
class Profile:
    name: str
    mass: float                  # kg
    inertia: tuple               # (Jx, Jy, Jz) kg m^2, diagonal
    rotors: tuple                # ((x, y), ...) m, body FLU
    spin: tuple                  # yaw torque sign of each rotor
    kf: float                    # N / (rad/s)^2
    km: float                    # N m / (rad/s)^2
    omega_max: float             # rad/s
    tau_up: float                # motor time constant, spinning up, s
    tau_down: float              # motor time constant, spinning down, s
    c_xy: float                  # rotor drag, N / (m/s) per rad/s of summed rotor speed
    c_z: float
    prop_radius: float           # m
    radius: float                # collision radius, m
    fm: float                    # figure of merit (est.)
    eta: float                   # electrical efficiency (est.)
    vmax_xy: float               # m/s
    vmax_up: float
    vmax_down: float
    tilt_max: float              # rad
    yaw_rate_max: float          # rad/s
    # autopilot gains (tuned by tests/test_physics.py)
    kp_pos: float                # 1/s
    kp_vel: float                # 1/s
    ki_vel: float                # 1/s^2
    kp_att: float                # 1/s
    kp_rate: tuple               # 1/s, roll pitch yaw
    ki_rate: tuple               # 1/s^2

    @property
    def hover_omega(self) -> float:
        return math.sqrt(self.mass * G / (4 * self.kf))

    def power(self, thrust):
        """Electrical power for rotor thrusts (momentum theory); sums over the last axis."""
        k = self.fm * self.eta * math.sqrt(2 * RHO * math.pi * self.prop_radius ** 2)
        return (thrust ** 1.5).sum(-1) / k

    @property
    def hover_power(self) -> float:
        return float(self.power(np.full(4, self.mass * G / 4)))

# gym-pybullet-drones assets/cf2x.urdf and cf2x.urdf constants (kf 3.16e-10 N/rpm^2, km 7.94e-12 N m/rpm^2, thrust2weight 2.25)
CF2 = Profile(
    name="cf2", mass=0.027, inertia=(1.4e-5, 1.4e-5, 2.17e-5),
    rotors=((0.028, -0.028), (-0.028, -0.028), (-0.028, 0.028), (0.028, 0.028)), spin=(-1, 1, -1, 1),
    kf=2.8816e-8, km=7.2404e-10, omega_max=2273.8, tau_up=0.02, tau_down=0.02,  # tau: est.
    c_xy=9.1785e-7, c_z=1.0311e-6, prop_radius=0.0231348, radius=0.06, fm=0.3, eta=0.5,
    vmax_xy=2.0, vmax_up=1.0, vmax_down=0.7, tilt_max=math.radians(25), yaw_rate_max=math.pi,
    kp_pos=1.0, kp_vel=2.0, ki_vel=0.5, kp_att=8.0, kp_rate=(15.0, 15.0, 8.0), ki_rate=(10.0, 10.0, 5.0),
)

# PX4 SITL gazebo-classic models/iris/iris.sdf.jinja (mass 1.5 + IMU 0.015 + 4 x 0.005 rotors); PX4 default speed limits
IRIS = Profile(
    name="iris", mass=1.535, inertia=(0.029125, 0.029125, 0.055225),
    rotors=((0.13, -0.22), (-0.13, 0.20), (0.13, 0.22), (-0.13, -0.20)), spin=(-1, -1, 1, 1),
    kf=5.84e-6, km=3.504e-7, omega_max=1100.0, tau_up=0.0125, tau_down=0.025,
    c_xy=1.75e-4, c_z=0.0, prop_radius=0.128, radius=0.384, fm=0.6, eta=0.8,
    vmax_xy=12.0, vmax_up=3.0, vmax_down=1.5, tilt_max=math.radians(45), yaw_rate_max=math.pi / 2,
    kp_pos=1.0, kp_vel=2.0, ki_vel=1.0, kp_att=8.0, kp_rate=(15.0, 15.0, 8.0), ki_rate=(10.0, 10.0, 5.0),
)

PROFILES = {"cf2": CF2, "iris": IRIS}

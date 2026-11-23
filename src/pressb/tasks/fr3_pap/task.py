"""Measured cube containment and support in an open tapered bowl, in metres."""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import math
import numpy as np
from scipy.spatial.transform import Rotation

INSTRUCTION = "Pick up blue cube and place it in the bowl."
DEFAULT_OUTCOME = dict(stable_seconds=.5, containment_tolerance_m=.002,
    max_linear_speed_m_s=.025, max_angular_speed_rad_s=.3,
    release_margin_m=.005, min_contact_force_n=.02, drop_below_table_m=.05,
    max_support_normal_tilt_deg=80., contact_separation_tolerance_m=.003,
    velocity_source='pose_delta', bowl_support_source='contact_force')


def number(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value


def outcome_config(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value)-set(DEFAULT_OUTCOME):
        raise ValueError("Unknown FR3 bowl outcome settings")
    settings = {**DEFAULT_OUTCOME, **value}
    choices = {'velocity_source': ('pose_delta', 'rigid_body'),
               'bowl_support_source': ('contact_force', 'vertical_force')}
    for key, allowed in choices.items():
        if settings[key] not in allowed:
            raise ValueError(f'{key} must be one of {allowed}')
    result = {key: v if key in choices else number(v, key) for key, v in settings.items()}
    if result['max_support_normal_tilt_deg'] >= 90:
        raise ValueError("Support tilt threshold must be below 90 degrees")
    return result


class BowlOutcome:
    """Success needs measured bowl support, cavity containment and released dwell.

    A cube may protrude above the rim while resting inside a shallow bowl. Below
    the rim all eight surface corners must fit its tapered cavity; above the rim
    they must remain within the mouth's vertical projection. A nearby cube on the
    table, a hovering held cube and an object perched on the outer rim fail.
    """
    def __init__(self, edge_m, table_top_z, bowl, **settings):
        self.edge = number(edge_m, 'edge_m')
        self.table = float(table_top_z)
        if not math.isfinite(self.table):
            raise ValueError('table_top_z must be finite')
        self.bowl = deepcopy(bowl)
        self.xy = array(bowl['center_xy'], (2,), 'bowl center_xy')
        self.height = number(bowl['height_m'], 'bowl height')
        self.bottom = self.table + number(bowl['bottom_thickness_m'], 'bowl bottom thickness')
        self.rim = self.table + self.height
        wall = number(bowl['wall_thickness_m'], 'bowl wall thickness')
        self.lower_radius = number(bowl['bottom_diameter_m'], 'bowl bottom diameter')/2-wall
        self.upper_radius = number(bowl['top_diameter_m'], 'bowl top diameter')/2-wall
        if not 0 < self.lower_radius < self.upper_radius or not self.bottom < self.rim:
            raise ValueError('Bowl must have a positive, widening inner cavity')
        self.vertices = np.array(list(product((-.5, .5), repeat=3))) * self.edge
        self.config = outcome_config(settings)
        self.previous_seconds = self.stable_since = self.first_success_time_s = None
        self.previous_position = self.previous_rotation = None
        self.ever_success = False
        self.max_lift_m = 0.

    def update(self, *, seconds, position, quaternion_wxyz, velocity, gripper_width_m, contacts):
        seconds = number(seconds, 'seconds', zero=True)
        if self.previous_seconds is not None and seconds <= self.previous_seconds:
            raise ValueError('Episode time must increase strictly')
        position = array(position, (3,), 'position')
        q = array(quaternion_wxyz, (4,), 'quaternion_wxyz')
        if abs(float(np.linalg.norm(q))-1) > .01:
            raise ValueError('Object quaternion must be normalized')
        velocity = array(velocity, (6,), 'velocity')
        width = number(gripper_width_m, 'gripper_width_m', zero=True)
        keys = {'finger_forces_n', 'bowl_support_force_n', 'bowl_contact_force_n', 'table_force_n'}
        if not isinstance(contacts, dict) or set(contacts)-keys:
            raise ValueError('Unknown FR3 bowl contact fields')
        fingers = array(contacts.get('finger_forces_n', [0., 0.]), (2,), 'finger forces')
        if np.any(fingers < 0):
            raise ValueError('Finger force magnitudes must be nonnegative')
        support = number(contacts.get('bowl_support_force_n', 0.), 'bowl support', zero=True)
        contact_force = number(contacts.get('bowl_contact_force_n', support), 'bowl contact', zero=True)
        table_force = number(contacts.get('table_force_n', 0.), 'table force', zero=True)
        orientation = Rotation.from_quat(q[[1, 2, 3, 0]])
        points = self.vertices @ orientation.as_matrix().T + position
        bottom, top = float(points[:, 2].min()), float(points[:, 2].max())
        cfg = self.config
        allowed_radius = self.lower_radius + np.clip((points[:, 2]-self.bottom)/(self.rim-self.bottom), 0., 1.) * (self.upper_radius-self.lower_radius)
        radii = np.linalg.norm(points[:, :2]-self.xy, axis=1)
        contained = bool(np.all(radii <= allowed_radius+cfg['containment_tolerance_m']) and
            bottom >= self.bottom-cfg['containment_tolerance_m'] and bottom < self.rim-cfg['containment_tolerance_m'] and
            position[2] <= self.rim+cfg['containment_tolerance_m'])
        # An object wedged inside a tapered bowl can be supported by friction
        # against near-horizontal contact normals. Vertical normal impulse
        # alone omits this support; retain it as an explicit legacy diagnostic.
        measured_support = contact_force if cfg['bowl_support_source'] == 'contact_force' else support
        supported = bool(measured_support >= cfg['min_contact_force_n'] and table_force < cfg['min_contact_force_n'])
        released = bool(width >= self.edge+cfg['release_margin_m'] and np.all(fingers < cfg['min_contact_force_n']))
        rigid_linear = float(np.linalg.norm(velocity[:3]))
        rigid_angular = float(np.linalg.norm(velocity[3:]))
        linear, angular = rigid_linear, rigid_angular
        if cfg['velocity_source'] == 'pose_delta' and self.previous_seconds is not None:
            dt = seconds-self.previous_seconds
            linear = float(np.linalg.norm(position-self.previous_position)/dt)
            angular = float((orientation*self.previous_rotation.inv()).magnitude()/dt)
        # GPU TGS + compliant foam can report nonzero solver velocities while
        # consecutive integrated poses are stationary. Judge actual motion at
        # each physics tick, keeping the same speed limits and continuous dwell.
        slow = bool(linear <= cfg['max_linear_speed_m_s'] and angular <= cfg['max_angular_speed_rad_s'])
        self.previous_seconds, self.previous_position, self.previous_rotation = seconds, position.copy(), orientation
        dropped = bool(top < self.table-cfg['drop_below_table_m'])
        self.max_lift_m = max(self.max_lift_m, bottom-self.table)
        valid = contained and supported and released and slow and not dropped
        if not valid:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = seconds
        stable = 0. if self.stable_since is None else seconds-self.stable_since
        success = bool(valid and stable+1e-12 >= cfg['stable_seconds'])
        event = success and not self.ever_success
        if event:
            self.ever_success, self.first_success_time_s = True, seconds
        return dict(success=success, success_event=bool(event), ever_success=self.ever_success,
            dropped=dropped, contained=contained, supported=supported, released=released,
            low_velocity=slow, bottom_world_m=bottom, top_world_m=top, max_lift_m=float(self.max_lift_m),
            stable_seconds=float(stable), first_success_time_s=self.first_success_time_s,
            linear_speed_m_s=linear, angular_speed_rad_s=angular,
            rigid_body_linear_speed_m_s=rigid_linear, rigid_body_angular_speed_rad_s=rigid_angular,
            velocity_source=cfg['velocity_source'], support_source=cfg['bowl_support_source'],
            support_force_n=measured_support,
            stability_blockers=[name for name, ok in [('containment', contained), ('support', supported),
                ('release', released), ('velocity', slow), ('not_dropped', not dropped)] if not ok])

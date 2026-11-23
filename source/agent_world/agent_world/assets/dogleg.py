# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
# =============================================================================

"""Isaac Lab asset configuration for the dogleg MJCF conversion.

The USD asset was imported from ``dogleg_floating.xml``.  Its equality
connect elements are represented as PhysX spherical joints with
``physics:excludeFromArticulation`` so the loops constrain the mechanism while
the main articulation remains a tree.  The values below are initial
simulation values since the source model does not contain motor limits or
actuator gains.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg

from agent_world import AssetPath


# Closed-chain neutral pose solved from dogleg_floating.xml so the wheel center
# matches the V14.2 neutral sagittal leg vector (0.1250 m, 3.03 deg).
# The joint angles are in the MJCF/PhysX revolute coordinates; right is mirrored.
DOGLEG_NEUTRAL_JOINT_POS = {
    **{f"chassis_roller_{index}_joint": 0.0 for index in range(1, 9)},
    "feeder_joint": 0.0,
    "left_wheel_joint": 0.0,
    "right_wheel_joint": 0.0,
    "gimbal_yaw_joint": 0.0,
    "pitch_drive_joint": 0.0,
    "pitch_crank_rod_joint": 0.0,
    "gimbal_pitch_joint": 0.0,
    "left_front1_joint": math.radians(25.849201682990344),
    "right_front1_joint": math.radians(-25.849201682990717),
    "left_front2_joint": math.radians(-39.746203069767006),
    "right_front2_joint": math.radians(39.74620306976585),
    "left_front3_joint": math.radians(-39.746203069896055),
    "right_front3_joint": math.radians(39.74620307004365),
    "left_front4_joint": math.radians(39.74620306987425),
    "right_front4_joint": math.radians(-39.74620306994898),
    "left_rear1_joint": math.radians(-36.96964323778274),
    "right_rear1_joint": math.radians(36.96964323791972),
    "left_rear2_joint": math.radians(39.746203069888935),
    "right_rear2_joint": math.radians(-39.7462030699833),
}


Dogleg_CFG = ArticulationCfg(
    soft_joint_pos_limit_factor=0.95,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/dogleg_floating/dogleg_floating.usd",
        activate_contact_sensors=True,
        copy_from_source=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=6,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos=DOGLEG_NEUTRAL_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Four leg drive joints.  The remaining leg joints are passive
        # mechanism joints.
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=[".*_front1_joint", ".*_rear1_joint"],
            stiffness=60.0,
            damping=2.0,
            effort_limit=40.0,
            velocity_limit=17.0,
            armature=0.0158,
        ),
        "legs_inact": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_rear2_joint",
            ],
            stiffness=0.0,
            damping=0.01,
            effort_limit=50.0,
            velocity_limit=300.0,
            armature=0.0001,
        ),
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=0.2,
            effort_limit=5.0,
            velocity_limit=60.0,
            armature=0.0,
        ),
        "gimbal_yaw": IdealPDActuatorCfg(
            joint_names_expr=["gimbal_yaw_joint"],
            stiffness=0.0,
            damping=0.5,
            effort_limit=2.0,
            velocity_limit=30.0,
            armature=0.0001,
        ),
        "gimbal_pitch": IdealPDActuatorCfg(
            joint_names_expr=["gimbal_pitch_joint"],
            stiffness=20.0,
            damping=0.5,
            effort_limit=10.0,
            velocity_limit=30.0,
            armature=0.0001,
        ),
        # Free mechanism joints from the XML.  They are left unpowered while
        # the spherical closure joints enforce the parallel-link geometry.
        "passive": IdealPDActuatorCfg(
            joint_names_expr=[
                "chassis_roller_.*_joint",
                "feeder_joint",
                "pitch_drive_joint",
                "pitch_crank_rod_joint",
            ],
            stiffness=0.0,
            damping=0.01,
            effort_limit=0.0,
            velocity_limit=300.0,
            armature=0.0001,
        ),
    },
)


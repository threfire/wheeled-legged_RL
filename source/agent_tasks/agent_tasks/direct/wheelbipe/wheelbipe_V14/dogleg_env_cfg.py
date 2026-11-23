"""Task configuration for the converted dogleg robot."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import agent_tasks.manager.mdp.isaaclab as mdp
from agent_world.assets.dogleg import Dogleg_CFG

from .cfg_utils import V14_ORDERED_LEG_BODY_NAMES, V14_ORDERED_LEG_JOINT_NAMES
from .env_cfg import EventCfgV14, WheelbipeV14FlatEnvCfg


@configclass
class DoglegEventCfg(EventCfgV14):
    """Domain randomization terms that match the converted asset."""

    guide_material = None
    spring_damping = None
    push_robot = None
    base_external_force_torque_xyz = None
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_vel_b,
        mode="reset",
        params={
            "pose_range": {"roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
        },
    )
    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    ".*_front1_link",
                    ".*_front2_link",
                    ".*_front3_link",
                    ".*_front4_link",
                    ".*_rear1_link",
                    ".*_rear2_link",
                    "gimbal_yaw_link",
                    "gimbal_pitch_link",
                ],
            ),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    legs_inact_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_rear2_joint",
                    ".*_front2_joint",
                    ".*_front3_joint",
                    ".*_front4_joint",
                ],
            ),
            "static_friction_distribution_params": (0.05, 0.1),
            "viscous_friction_distribution_params": (0.01, 0.025),
            "operation": "add",
            "distribution": "uniform",
        },
    )
@configclass
class DoglegFlatEnvCfg(WheelbipeV14FlatEnvCfg):
    """Flat terrain configuration using the converted dogleg asset."""

    def __post_init__(self):
        super().__post_init__()
        # 8192 environments required ~1.56M patches. Allow headroom for 16384.
        # Isaac Lab's default (163840) is too small for large Dogleg batches.
        self.sim.physx.gpu_max_rigid_patch_count = 2**22

    robot_cfg = Dogleg_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()
    # The inherited V14 reset IK does not close Dogleg's loops yet. Keep
    # random IK reset disabled until the Dogleg-specific solver is calibrated.
    use_leg_random_start = False
    events = DoglegEventCfg()
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    legs_act_name = [".*_front1_joint", ".*_rear1_joint"]
    legs_inact_name = [
        ".*_front2_joint",
        ".*_front3_joint",
        ".*_front4_joint",
        ".*_rear2_joint",
    ]
    wheel_name = [".*_wheel_joint"]
    spring_name = []
    use_spring = False

    # Match V14's policy target clipping (about +/-180 degrees). These are
    # command bounds around the configured neutral pose, not PhysX stops.
    lower_joint_limit = -3.14
    upper_joint_limit = 3.14
    leg_target_limits_relative_to_default = True

    gimbal_yaw_name = ["gimbal_yaw_joint"]
    gimbal_pitch_name = ["gimbal_pitch_joint"]
    use_gimbal = True
    gimbal_pitch_target_pos = 0.0
    gimbal_yaw_velocity_range = (0.0, 0.0)
    gimbal_heading_control_cfg = {
        "enabled": True,
        "target_mode": "fixed",
        "fixed_heading": 0.0,
        "kp": 20.0,
        "kd": 1.0,
        "randomize_gains": False,
        "max_effort": 5.0,
        "apply_only_in_special_mode": False,
    }

    ordered_leg_joint_names = V14_ORDERED_LEG_JOINT_NAMES
    ordered_leg_body_names = V14_ORDERED_LEG_BODY_NAMES
    mute_wheel_pos_obs = True


@configclass
class DoglegStandEnvCfg(DoglegFlatEnvCfg):
    """Stand-up and balance stage with zero locomotion commands."""

    use_leg_random_start = False
    use_predefined_leg_random_start = False
    use_joint_vel_random_start = False

    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.ranges.heading = (0.0, 0.0)


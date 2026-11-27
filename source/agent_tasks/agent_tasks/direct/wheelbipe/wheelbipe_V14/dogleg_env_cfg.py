"""Task configuration for the converted dogleg robot."""

import copy

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import agent_tasks.manager.mdp.isaaclab as mdp
from agent_world.assets.dogleg import Dogleg_CFG, DoglegLegacy_CFG

from .cfg_utils import V14_ORDERED_LEG_BODY_NAMES, V14_ORDERED_LEG_JOINT_NAMES
from .env_cfg import EventCfgV14, WheelbipeV14FlatEnvCfg, WheelbipeV14RoughEnvCfg_v1


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
        # Keep the tested 20000-env run within the 2**22 contact-patch buffer.
        self.sim.physx.gpu_max_rigid_patch_count = 2**22
        # Dogleg reset IK targets use a wider validated length interval than
        # the inherited V14 leg geometry.
        self.leg_length_range = [0.13, 0.40]
        self.predefined_reset_ground = copy.deepcopy(self.predefined_reset_ground)
        self.predefined_reset_ground["leg_length"] = [0.13, 0.40]
        for mode in self.predefined_reset_ground.get("modes", {}).values():
            mode["leg_length"] = [0.13, 0.40]

    robot_cfg = Dogleg_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()
    # Dogleg's own closed-chain IK is used by the inherited random reset path.
    use_leg_random_start = True
    # Randomize target poses, but hold reset joint velocities at zero.
    use_joint_vel_random_start = False
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
        # Strengthen the zero-command drift penalty for Dogleg stand fine-tuning.
        self.rewards["stand_still_lin_vel"] = -2.0


@configclass
class DoglegUnifiedEnvCfg(DoglegFlatEnvCfg):
    """Mixed zero/non-zero velocity commands for a unified locomotion policy."""

    def __post_init__(self):
        # Keep DoglegFlatEnvCfg's command sampler: it includes standing
        # environments plus forward/reverse and turning commands.  Allocate
        # one tenth of environments to zero-velocity commands while retaining
        # the base V14 zero-command drift penalty.
        super().__post_init__()
        self.commands.rel_standing_envs = 0.1
        self.rewards["stand_still_lin_vel"] = -1.0


@configclass
class DoglegUnifiedEnvCfgV1(DoglegUnifiedEnvCfg):
    """Unified Flat-v1: retain v0 commands and add gated recovery training."""

    robot_cfg = Dogleg_CFG.replace(
        spawn=Dogleg_CFG.spawn.replace(
            usd_path=Dogleg_CFG.spawn.usd_path.replace(
                "dogleg_floating_physics_contactfix.usda",
                "dogleg_floating_physics_compound_candidate.usda",
            )
        ),
        prim_path="/World/envs/env_.*/Robot",
    ).copy()

    recovery_training_cfg = {
        "enabled": True,
        "iteration_start": 1000,
        "fallen_pitch_gravity_x_min": 0.75,
        "fallen_inverted_gravity_z_min": 0.55,
        "upright_projected_gravity_xy_max": 0.35,
        "upright_projected_gravity_z_max": -0.80,
        "minimum_root_height": 0.20,
        "maximum_root_linear_speed": 0.35,
        "maximum_root_angular_speed": 0.60,
        "stable_duration_s": 0.50,
        "height_target": 0.30,
    }
    # Forward/backward falls only, totaling 10% of reset samples from iteration
    # 1000 onward. Lateral roll starts are intentionally excluded.
    predefined_reset_air = {
        "enabled": True,
        "modes": (
            {
                "name": "recover_forward",
                "recover": True,
                "prob": 0.05,
                "iteration_start": 1000,
                "iteration_end": -1,
                "pose_range": {"z": (-0.24, -0.18), "roll": (-0.15, 0.15), "pitch": (1.20, 1.90), "yaw": (-torch.pi, torch.pi)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "leg_length_range": (0.18, 0.32),
                "leg_angle_range": (-0.30, 0.30),
                "command_limits": {"duration_s": 60.0, "lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0), "height": (0.30, 0.30)},
            },
            {
                "name": "recover_backward",
                "recover": True,
                "prob": 0.05,
                "iteration_start": 1000,
                "iteration_end": -1,
                "pose_range": {"z": (-0.24, -0.18), "roll": (-0.15, 0.15), "pitch": (-1.90, -1.20), "yaw": (-torch.pi, torch.pi)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "leg_length_range": (0.18, 0.32),
                "leg_angle_range": (-0.30, 0.30),
                "command_limits": {"duration_s": 60.0, "lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0), "height": (0.30, 0.30)},
            },
        ),
    }

    def __post_init__(self):
        super().__post_init__()
        # Restore the ordinary command range used by Unified Flat-v0 / V14.
        self.commands.ranges.lin_vel_x = (-2.7, 2.7)
        # Keep the original Dogleg Flat height-command distribution unchanged.
        self.height_range = [0.20, 0.42]
        self.rewards["recovery_upright"] = 2.0
        self.rewards["recovery_height"] = 1.0


@configclass
class DoglegRecoveryBoxCandidateEnvCfg(DoglegUnifiedEnvCfgV1):
    """GUI trial: compound-box candidate and 50% forward/backward recovery starts."""

    robot_cfg = Dogleg_CFG.replace(
        spawn=Dogleg_CFG.spawn.replace(
            usd_path=Dogleg_CFG.spawn.usd_path.replace(
                "dogleg_floating_physics_contactfix.usda",
                "dogleg_floating_physics_compound_candidate.usda",
            )
        ),
        prim_path="/World/envs/env_.*/Robot",
    ).copy()

    def __post_init__(self):
        super().__post_init__()
        self.recovery_training_cfg = copy.deepcopy(self.recovery_training_cfg)
        self.recovery_training_cfg["iteration_start"] = 0
        self.predefined_reset_air = copy.deepcopy(self.predefined_reset_air)
        self.predefined_reset_air["modes"] = tuple(
            {**mode, "prob": 0.25, "iteration_start": 0}
            for mode in self.predefined_reset_air["modes"]
        )


@configclass
class DoglegUnifiedLegacyAssetEnvCfg(DoglegUnifiedEnvCfg):
    """Unified command task using the original, unmodified Dogleg USD."""

    robot_cfg = DoglegLegacy_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()


@configclass
class DoglegRoughEnvCfg_v1(WheelbipeV14RoughEnvCfg_v1):
    """V14 Rough-v1 terrain/state machines with Dogleg's tested robot setup.

    Only robot-specific settings are copied from Flat. Rough commands, terrain,
    reset modes, observations and rewards remain those of Rough-v1, except for
    the explicitly adapted geometry-dependent reward terms below.
    """

    def __post_init__(self):
        super().__post_init__()
        dogleg = DoglegFlatEnvCfg()
        for name in (
            "robot_cfg", "events", "contact_sensor",
            "use_leg_random_start", "use_joint_vel_random_start",
            "legs_act_name", "legs_inact_name", "wheel_name",
            "spring_name", "use_spring",
            "lower_joint_limit", "upper_joint_limit", "leg_target_limits_relative_to_default",
            "gimbal_yaw_name", "gimbal_pitch_name", "use_gimbal",
            "gimbal_pitch_target_pos", "gimbal_yaw_velocity_range", "gimbal_heading_control_cfg",
            "ordered_leg_joint_names", "ordered_leg_body_names", "mute_wheel_pos_obs",
        ):
            setattr(self, name, copy.deepcopy(getattr(dogleg, name)))
        self.sim.physx.gpu_max_rigid_patch_count = dogleg.sim.physx.gpu_max_rigid_patch_count
        # 20000 robots on Rough terrain overflowed the default 64 MiB stack
        # during startup (PhysX requested at least 73,145,448 bytes). Reserve
        # headroom for denser terrain contacts; this is separate from patches.
        self.sim.physx.gpu_collision_stack_size = 2**28  # 256 MiB

        # The imported CAD hierarchy has an extra base_link container.
        self.height_scanner.prim_path = "/World/envs/env_.*/Robot/base_link/base_link"
        self.left_wheel_height_scanner.prim_path = "/World/envs/env_.*/Robot/base_link/left_wheel_link"
        self.right_wheel_height_scanner.prim_path = "/World/envs/env_.*/Robot/base_link/right_wheel_link"

        # V14's rear2-rear1 numeric interval is tied to its own joint zero/axes.
        # Do not apply it to the mirrored Dogleg CAD joint coordinates.
        self.front_rear_joint_limit_rewards_enabled = False
        for name in (
            "rear2_rear1_joint_pos_limits", "rear2_rear1_joint_pos_limits_torque",
            "rear2_rear1_joint_pos_limits_vel", "airborne_joint_pos_limits",
            "airborne_joint_pos_limits_vel_reg",
        ):
            self.rewards[name] = 0.0
            self.airborne_state_machine_cfg["reward_additions"].pop(name, None)

        # Positive vehicle motion requires negative left / positive right raw
        # joint velocity. This mapping affects only airborne wheel-speed rewards.
        for name in (
            "airborne_precontact_wheel_directional_speed",
            "airborne_precontact_wheel_directional_speed_shortfall",
        ):
            self.airborne_state_machine_cfg["reward_additions"][name]["joint_velocity_signs"] = {
                "left_wheel_joint": -1.0,
                "right_wheel_joint": 1.0,
            }


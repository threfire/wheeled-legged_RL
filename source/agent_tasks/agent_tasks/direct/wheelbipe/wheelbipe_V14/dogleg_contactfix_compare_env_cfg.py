"""Isolated asset-comparison configuration for the Dogleg Unified Flat task.

This module inherits every setting from the current Unified Flat-v1 config and
changes only the robot USD to the former contactfix overlay used by Dogleg
Rough training. The production v1 configuration is intentionally untouched.
"""

from isaaclab.utils import configclass

from agent_world.assets.dogleg import Dogleg_CFG

from .dogleg_env_cfg import DoglegUnifiedEnvCfgV1


@configclass
class DoglegUnifiedContactfixCompareEnvCfg(DoglegUnifiedEnvCfgV1):
    """Compare the half-mesh/half-box contactfix asset against compound boxes."""

    robot_cfg = Dogleg_CFG.replace(
        spawn=Dogleg_CFG.spawn.replace(
            usd_path=Dogleg_CFG.spawn.usd_path.replace(
                "dogleg_floating_physics_compound_candidate.usda",
                "dogleg_floating_physics_contactfix.usda",
            )
        ),
        prim_path="/World/envs/env_.*/Robot",
    ).copy()

# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

import gymnasium as gym

from agent_tasks.direct.wheelbipe import agents

# Isolated collision-asset comparison task; its configuration lives in a new
# module so the production Dogleg Unified-v1 source remains unchanged.
from . import dogleg_contactfix_compare  # noqa: F401

gym.register(
    id="Robotics-Dogleg-Rough-v1",
    entry_point=f"{__name__}.dogleg_env:DoglegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dogleg_env_cfg:DoglegRoughEnvCfg_v1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoglegRoughPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Dogleg-Flat-v0",
    entry_point=f"{__name__}.dogleg_env:DoglegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dogleg_env_cfg:DoglegFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoglegFlatPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Dogleg-Stand-Flat-v0",
    entry_point=f"{__name__}.dogleg_env:DoglegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dogleg_env_cfg:DoglegStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoglegFlatPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Dogleg-Unified-Flat-v0",
    entry_point=f"{__name__}.dogleg_env:DoglegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dogleg_env_cfg:DoglegUnifiedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoglegFlatPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Dogleg-Unified-Flat-v1",
    entry_point=f"{__name__}.dogleg_env:DoglegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dogleg_env_cfg:DoglegUnifiedEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoglegFlatPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Dogleg-Recovery-Box-Candidate-v0",
    entry_point=f"{__name__}.dogleg_env:DoglegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dogleg_env_cfg:DoglegRecoveryBoxCandidateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoglegFlatPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Dogleg-Unified-Legacy-Asset-v0",
    entry_point=f"{__name__}.dogleg_env:DoglegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dogleg_env_cfg:DoglegUnifiedLegacyAssetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoglegFlatPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Wheelbipe-V14-Flat-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-v1",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatEnvCfg_v1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-v2",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatEnvCfg_v2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-Play-v2",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatEnvCfg_v2_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Rough-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14RoughPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Rough-v1",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14RoughEnvCfg_v1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14RoughPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-DreamWaQ-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatDreamWaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatDreamWaqPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-HIM-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatHIMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatHIMPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-NP3OBarlow-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatNP3OBarlowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatNP3OBarlowPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-Play-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-DreamWaQ-Play-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatDreamWaqEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatDreamWaqPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-HIM-Play-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatHIMEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatHIMPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Flat-NP3OBarlow-Play-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14FlatNP3OBarlowEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14FlatNP3OBarlowPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Rough-Play-v0",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14RoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14RoughPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheelbipe-V14-Rough-Play-v1",
    entry_point=f"{__name__}.env:WheelbipeV14Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeV14RoughEnvCfg_v1_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeV14RoughPPORunnerCfg",
    },
)


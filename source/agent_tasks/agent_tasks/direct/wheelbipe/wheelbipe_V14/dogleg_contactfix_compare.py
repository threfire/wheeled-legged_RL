"""Registration for the isolated Dogleg collision-asset comparison task."""

import gymnasium as gym

from agent_tasks.direct.wheelbipe import agents


gym.register(
    id="Robotics-Dogleg-Unified-Contactfix-Compare-v0",
    entry_point="agent_tasks.direct.wheelbipe.wheelbipe_V14.dogleg_env:DoglegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "agent_tasks.direct.wheelbipe.wheelbipe_V14."
            "dogleg_contactfix_compare_env_cfg:DoglegUnifiedContactfixCompareEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoglegFlatPPORunnerCfg",
    },
)

# Dogleg 训练与仿真启动手册

本文整理当前 Dogleg 统一策略的 Isaac Lab 训练、TensorBoard、MuJoCo sim2sim 和键盘控制命令。当前对照实验使用 **512 个并行环境、图形界面、零速命令比例 0、CUDA 0**，从随机初始化开始，不加载任何 Dogleg checkpoint，也不恢复旧优化器状态；学习率调度恢复为 V14 的 adaptive 设置。

## 1. Isaac Lab 训练

当前默认启动脚本已设为图形界面、512 环境、零速命令比例 `0`、从随机初始化开始，定点漂移惩罚权重为 `-1.0`。直接运行：

```bash
cd /root/gpufree-data/wheeled-legged_RL
./scripts/start_dogleg_unified_training.sh
```

脚本顶部的 `NUM_ENVS`、`ZERO_COMMAND_ENV_FRACTION`、`STAND_STILL_LIN_VEL_WEIGHT`、`DEVICE` 和 `HEADLESS` 可直接修改。

先进入训练仓库并设置 Python 路径：

```bash
cd /root/gpufree-data/wheeled-legged_RL
export TERM=xterm-256color
export PYTHONPATH="$PWD/source/agent_world:$PWD/source/agent_tasks:$PWD/source/agent_rl:$PWD"
```

### 带图形界面训练

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task=Robotics-Dogleg-Unified-Flat-v0 \
  --num_envs=512 \
  --max_iterations=20000 \
  --zero_command_env_fraction=0.0 \
  --stand_still_lin_vel_weight=-1.0 \
  --run_name=dogleg_unified_scratch_512_gui_zero0 \
  --checkpoint_tag=scratch \
  --device=cuda:0
```

省略 `--headless` 会打开 Isaac Sim 窗口。图形渲染会占用额外资源；训练窗口中机器人很多时，可以缩小视口或降低渲染质量。

### 无图形界面训练

使用同一条命令并加上 `--headless`：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task=Robotics-Dogleg-Unified-Flat-v0 \
  --num_envs=512 \
  --max_iterations=20000 \
  --zero_command_env_fraction=0.0 \
  --stand_still_lin_vel_weight=-1.0 \
  --run_name=dogleg_unified_scratch_512_headless_zero0 \
  --checkpoint_tag=scratch \
  --headless \
  --device=cuda:0
```

### 常改参数

| 参数 | 当前值 | 用途 |
| --- | ---: | --- |
| `--num_envs` | `512` | 并行环境数；按资源情况调整 |
| `--max_iterations` | `20000` | 最大训练迭代数；checkpoint 每 500 轮保存一次 |
| `--zero_command_env_fraction` | `0.0` | 当前先训练非零速度移动；之后再增加零速比例 |
| `--stand_still_lin_vel_weight` | `-1.0` | 零速附近的平移漂移惩罚，必须为零或负数 |
| `--checkpoint` | 不传 | 策略随机初始化，不继承旧 Dogleg 权重 |
| `--run_name` | 依启动方式命名 | TensorBoard run 名和日志目录后缀 |
| `--checkpoint_tag` | `scratch` | checkpoint 短标签，产物如 `model_500_scratch.pt` |
| `--headless` | 默认启动脚本开启 | 加上后无图形运行，节省渲染资源 |

Dogleg runner 当前恢复 V14 PPO 设置：初始学习率 `1e-4`，`schedule=adaptive`。这组实验先令零速命令比例为 0，用来检查从随机初始化能否先学会平地移动。

本任务从 `DoglegFlatEnvCfg` 继承混合速度命令，并覆盖零速比例与定点漂移惩罚。不要改成 `Robotics-Dogleg-Stand-Flat-v0` 来训练统一策略；该任务把全部速度命令固定为零。

这次从随机初始化开始，因为之前的 Dogleg 策略是在碰撞体配置尚未校准时训练的。新的 run 名带 `scratch`，便于和旧实验区分。

每次新训练会创建新的时间戳日志目录，旧 checkpoint 和事件文件不会被覆盖。训练配置会写入新 run 的 `params/env.yaml` 和 `params/agent.yaml`，可以用来核对并行数、零速比例、奖励权重及实际加载的碰撞 USD。

## Dogleg Rough-v1

Rough-v1 的独立启动脚本、5000 IK checkpoint 继承方式和配置说明见 [DOGLEG_ROUGH_V1.md](DOGLEG_ROUGH_V1.md)。

```bash
cd /root/gpufree-data/wheeled-legged_RL
./scripts/start_dogleg_rough_v1_training.sh
```

## 2. 打开 TensorBoard

训练时另开一个终端：

```bash
cd /root/gpufree-data/wheeled-legged_RL
/workspace/isaaclab/_isaac_sim/python.sh -m tensorboard.main \
  --logdir="$PWD/logs/rsl_rl/dogleg_flat_direct" \
  --host=0.0.0.0 \
  --port=6006
```

在同一容器的浏览器打开：

```text
http://localhost:6006
```

远程桌面浏览器通常也可用 `http://127.0.0.1:6006`。若 6006 端口已被占用，将命令中的端口和网址同时改成其他值，例如 6007。

建议先看 `Episode/Reward/track_lin_vel_xy`、`Episode/Reward/track_ang_vel_z`、`Episode/Reward/track_height_exp_tight`、`Episode/Reward/track_height_square`、`Episode/Reward/stand_still_lin_vel`、`Train/mean_episode_length`、`Episode/Reset/terminate` 和 `Policy/mean_noise_std`。奖励项曲线是加权后的 episode 统计；更改权重后，比较定点项时要把曲线绝对值除以权重绝对值。

## 3. 导出 checkpoint 给 MuJoCo 使用

把 `CHECKPOINT` 改成要评估的模型，然后运行一次步进并立即退出。`play.py` 会在 checkpoint 同目录下创建 `exported/policy.onnx`：

```bash
cd /root/gpufree-data/wheeled-legged_RL
export TERM=xterm-256color
export PYTHONPATH="$PWD/source/agent_world:$PWD/source/agent_tasks:$PWD/source/agent_rl:$PWD"
RUN_DIR="$(find "$PWD/logs/rsl_rl/dogleg_flat_direct" -maxdepth 1 -type d -name '*dogleg_unified_scratch_512_gui_zero0' | sort | tail -n 1)"
CHECKPOINT="$RUN_DIR/model_500_scratch.pt"

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task=Robotics-Dogleg-Unified-Flat-v0 \
  --checkpoint="$CHECKPOINT" \
  --num_envs=1 \
  --max_steps=1 \
  --device=cpu
```

## 4. 启动 Dogleg MuJoCo sim2sim

以下脚本固定使用本次 IK 微调训练的 `2026-11-26_06-44-42_dogleg_contactfix_from1500_weights_20000_zero10_ik/model_1500_ik.pt` 导出策略，文件名为 `exported/model_1500_ik.onnx`。不是旧的 `model_1500_contactfix.pt`。导出目录的同名 JSON 记录源 checkpoint 路径、SHA256 和输出校验结果。运行部署不会恢复或中断 Isaac 的训练。

```bash
cd /root/gpufree-data/wheelbipe_ros2_sim2sim
./scripts/start_dogleg_sim1500.sh
```

脚本自动加载 ROS 环境，仿真默认自动进入 RL。如需无画面运行，可追加 `--headless`。如果部署仓库尚未构建，先运行 `./scripts/build.sh`。

## 5. 启动 ROS 2 键盘控制

终端一按上节启动 MuJoCo。另开终端启动原来的终端键盘控制：

```bash
cd /root/gpufree-data/wheelbipe_ros2_sim2sim
./scripts/start_dogleg_keyboard.sh
```

保持运行键盘节点的终端获得焦点，观看旁边的 MuJoCo 窗口。界面、状态切换和高度按键沿用原版，只修改 WASD 的速度赋值方式。不要同时运行多个键盘节点。

- 按住 `W/S` 加速前进/后退，按住 `A/D` 加速转向，允许同时按 `W+A` 等组合。
- 松开按键，速度命令快速衰减到零；相反方向键同时按下时，对应轴回零。
- `T/G` 按原方式步进调高/调低高度，松开保持；`R` 恢复默认高度。
- `0/1/2/3` 切换 INIT/IDLE/PREPARE/RL，`X` 退出。
- `空格` 立刻清零命令；在 X11 远程桌面下，终端失焦也清零速度。

默认线速度上限 `2.5 m/s`，角速度上限 `3 rad/s`，线加速度 `2 m/s²`，松开线减速度 `10 m/s²`（从满速约 0.25 秒回零）。归零的是目标命令，机器人实际制动仍取决于策略与动力学。

修改 `wheelbipe_ros2_sim2sim/src/tools/keyboard_teleop/config/keyboard_teleop_params.yaml` 的 `max_lin_vel_x`、`max_ang_vel_z`、`lin_vel_x_rate`、`lin_vel_x_decel` 等参数即可调整手感，也可以临时追加 ROS 参数覆盖，例如：

```bash
./scripts/start_dogleg_keyboard.sh -p max_lin_vel_x:=1.5 -p lin_vel_x_rate:=3.0 -p lin_vel_x_decel:=12.0
```

Dogleg reset IK samples leg-length targets over `0.13–0.40 m`, with the Dogleg IK lookup table validated over that interval. The height command is a different coordinate: world Z of the Dogleg `base_link` root origin. On flat ground with the wheel in contact, the wheel joint center is one wheel radius above ground, so `H_base = 0.06 + L*cos(theta)`. At the calibrated neutral leg angle (`theta ≈ -3.03°`), `L=0.13–0.40 m` maps to `H_base≈0.190–0.459 m`. Unified Flat-v1 retains the original `0.20–0.42 m` height-command range, and MuJoCo keyboard/controller use that same range. The mapping is exact only for the stated flat-ground, upright, neutral-angle reference; as the leg angle changes, the same root height corresponds to a different geometric leg length.

## 6. 验证碰撞体

Dogleg Flat-v0 与 Rough-v1 使用 `dogleg_floating_physics_contactfix.usda`。新的 Unified Flat-v1 使用细致复合盒体候选 `dogleg_floating_physics_compound_candidate.usda`：机身与 front2 连杆的单一粗盒替换成多个盒体，轮/导轮圆柱和 10 个闭链约束保留。V1 启动脚本会检查并打印实际 USD 路径。可在 Isaac Sim 的 Physics Debugger 打开 Colliders 显示；USD 改动不会热更新已经启动的训练进程，必须重启仿真/训练才能载入新资产。

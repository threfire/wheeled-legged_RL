# Dogleg Rough-v1

任务 ID：`Robotics-Dogleg-Rough-v1`；TensorBoard 分类：`dogleg_rough_v1`。

## 启动训练

```bash
cd /root/gpufree-data/wheeled-legged_RL
./scripts/start_dogleg_rough_v1_training.sh
```

默认配置：16384 个环境、GPU `cuda:0`、无图形界面、新训练 6000 轮。
脚本使用以下 checkpoint 初始化网络权重，**不恢复优化器，也不恢复 5000 轮计数**：

```text
logs/rsl_rl/dogleg_flat_direct/2026-11-26_06-44-42_dogleg_contactfix_from1500_weights_20000_zero10_ik/model_5000_ik.pt
```

输出在 `logs/rsl_rl/dogleg_rough_v1/<时间戳>_dogleg_rough_v1_from5000_weights_16384/`；checkpoint 如 `model_500_rough.pt`。
原来的 Flat 启动脚本仍用于 Flat 训练。

修改新脚本开头的 `NUM_ENVS`、`MAX_ITERATIONS`、`HEADLESS`、`CHECKPOINT` 等默认值即可调整。也可临时覆盖：

```bash
# 无图形界面训练，默认 16384 环境
./scripts/start_dogleg_rough_v1_training.sh

# 例如减少并行数，无图形训练
NUM_ENVS=8192 HEADLESS=true ./scripts/start_dogleg_rough_v1_training.sh
```

此前 20000 环境的无界面运行出现 PhysX 显存分配失败和 PyTorch OOM，现将默认并行数降为 16384，以留出物理计算和 PPO 更新所需的显存余量；16384 的稳定性需通过实际运行确认。

### 轻量预览窗口

当前默认关闭预览，20000 环境的可视化启动尚未验证通过。只有显式设置 `HEADLESS=false` 时才启用实验性预览：`performance` 渲染模式、`960×540` 视口，隐藏编辑器面板，相机跟随 `env_0` 的机器人。其他环境只设置 USD 可见性为隐藏，机器人、碰撞体、传感器和训练采样全部保留。地形保持显示。

脚本中的 `PREVIEW_RENDER_INTERVAL=20` 表示每 20 个物理步刷新一次画面，即每 0.1 秒仿真时间一帧；实际墙钟帧率取决于训练速度。物理步长仍是 `0.005 s`，策略步长仍是 `0.02 s`。渲染开销低于显示全部机器人的普通窗口，但仍高于纯无界面模式。

可修改 `PREVIEW_WIDTH`、`PREVIEW_HEIGHT`、`PREVIEW_RENDER_INTERVAL` 或 `PREVIEW_ENV_INDEX`。如需完全关闭窗口：

```bash
HEADLESS=true ./scripts/start_dogleg_rough_v1_training.sh
```

对已启动进程修改脚本不会生效，需要先停止再启动。

大量环境初始化期间，窗口可能暂时没有画面。预览现在会在物理初始化/首次渲染前就隐藏其他环境，且构建期间暂停视口更新。此前是在初始化之后才隐藏，20000 环境仍会先进入全部外观的渲染准备过程。Kit 默认的 120 秒卡顿检测也会在大规模克隆时误报；现在只在构建期间暂停检测，完成后恢复，避免启动过程中出现要求强制退出的弹框。

## 配置来源与适配

- 机器人沿用当前 Dogleg 的 `dogleg_floating_physics_contactfix.usda`，包括轮地碰撞、10 个闭链连接、质量惯量、中性零位、执行器参数与云台保持控制。
- 复用 `DoglegEnv` 和 Dogleg IK 查表；保留随机腿姿态重置、重置关节速度为零、无气弹簧及现有 Dogleg 随机化事件。
- 地形、按地形采样的命令、台阶/斜坡/腾空状态机、空中和地面重置、奖励取自本地 `WheelbipeV14RoughEnvCfg_v1`。包含台阶、坡面、平面、随机起伏等 11 类地形。
- 机体及左右轮的三个高度扫描器改为 Dogleg USD 的实际刚体路径。Rough 使用相对地面的高度，而非 Flat 的绝对世界高度。
- 停用 V14 专属的 `rear2−rear1` 角度区间惩罚，包括空中同类惩罚；该数值不能直接对应 Dogleg 的零位与左右关节轴。
- 空中落地前的轮速方向奖励按 Dogleg 的左右轮关节符号计算：前进对应左轮负、右轮正。策略动作和观测中的原始轮速符号不变；原 V14 默认奖励计算不变。
- Actor 输入 35 维、Critic 输入 78 维、动作 6 维，和此次 `model_5000_ik.pt` 一致，网络可以完整加载。
- 沿用 V14 PPO：学习率初值 `1e-4`、`adaptive` KL 调整。Rough 的高速/旋转命令按其自身配置从新轮次开始生效，不沿用 Flat 脚本的第 2000 轮高速门槛。

零速命令沿用 Rough-v1：`rel_standing_envs=0.1`，另有 Rough 的 `zero_cmd` 特殊模式和地形命令覆盖；不要把整体零速比例简单理解为严格固定 10%。站立速度惩罚仍为 `-1`。

地形生成器的 `curriculum=True` 表示按行生成不同难度；本任务保留 V14 原有的初始地形分配和课程开关，不额外启用按轮次推进的高度课程。
Rough 与 Flat 的奖励内容和任务分布不同，总奖励不能直接横比。关注速度跟踪、高度跟踪、提前终止，以及不同地形的实际通过表现。

## 查看曲线

```bash
cd /root/gpufree-data/wheeled-legged_RL
/workspace/isaaclab/_isaac_sim/python.sh -m tensorboard.main \
  --logdir="$PWD/logs/rsl_rl/dogleg_rough_v1" \
  --host=0.0.0.0 --port=6007
```

容器浏览器打开 `http://localhost:6007`。6007 可与原来 Flat 的 6006 同时使用。

## 验证记录

2026-11-26：通过实际训练入口完成 8 环境、2 轮 GPU PPO 验证，17 个网络张量全部加载，输入扩展和不兼容跳过均为 0；验证日志单独保存在 `logs/rsl_rl/dogleg_rough_v1_smoke/`，不混入正式训练曲线。

另完成配置一致性检查、左右轮方向奖励回归检查，以及 16 环境连续 256 步、8 次显式重置的运行检查。实际覆盖 8 个空中重置和 1805 个腾空环境步；观测、奖励、关节状态与三个扫描器命中点均为有限值。10 个闭链连接在重置后的最大锚点残差为 `0.0212 mm`。这些检查证明任务可运行，不代表已经学会上台阶或通过全部地形。

轻量预览已通过 8 环境、2 轮带 GUI 的 PPO 检查。另检查 4 个 PhysX 实例全部存在，`env_1..3` 的外观不可见且其根状态仍在步进中变化；`env_0` 正常渲染，视口截图为 `960×540`。物理/策略步长保持 `0.005/0.02 s`。

## 20000 环境启动时的碰撞缓冲区溢出

如果出现 `collisionStackSize buffer overflow` 和 `Contacts have been dropped`，代表碰撞临时计算容量不足，部分物理接触被丢弃。不要把受影响的运行作为正常训练结果。

已将 Rough 配置的 `sim.physx.gpu_collision_stack_size` 从默认 `2**26`（64 MiB）提高到 `2**28`（256 MiB）。首次 20000 环境运行报告需要至少 `73,145,448` 字节，新的容量高于该需求并保留余量；已有 `gpu_max_rigid_patch_count=2**22` 保持不变，两者是不同缓冲区。

修改后需要在原训练终端按 `Ctrl+C` 停止，再重新运行本页启动脚本。缓冲区在物理场景创建时分配，不能通过修改文件热更新正在运行的进程。此容量修复通过语法检查；20000 环境实际是否消除溢出，需要重启验证。

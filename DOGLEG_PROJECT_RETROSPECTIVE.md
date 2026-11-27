# Dogleg 项目复盘

记录 Dogleg 从模型转换、Isaac Lab 训练、碰撞体排查到 MuJoCo sim2sim 的关键过程，供后续复现实验和定位问题。本文区分已经由配置或数值测试证实的事实、历史实验观察和仍未验证的假设。

## 当前状态

- 当前训练任务是 `Robotics-Dogleg-Unified-Flat-v0`：Flat 平地任务继承 V14 Flat 配置，再覆盖 Dogleg 资产与 reset 行为。它不是直接使用 V13 任务；V14 自身继承了较底层的 V13/V25 通用环境代码。
- 当前正式物理资产由 `Dogleg_CFG` 加载 `dogleg_floating_physics_contactfix.usda`。它保留 10 个 PhysX 球形闭链约束，并对 base_link、左右 front2_link 使用恢复的原网格碰撞体。
- Dogleg reset 现在使用 Dogleg 专用 IK 查找表，目标范围为腿长 `0.13–0.36 m`、腿角 `−120°–135°`。Unified/Flat 随机姿态 reset 已开启；初始关节速度随机化关闭。单独的 `DoglegStandEnvCfg` 仍关闭随机腿姿态。
- 最新训练 run 是 `2026-11-26_06-44-42_dogleg_contactfix_from1500_weights_20000_zero10_ik`。它只继承 `model_1500_contactfix.pt` 的策略权重，优化器与计数从头开始；20,000 环境、零速占比 10%、上限 6,000 个新迭代、headless。核对时日志约在第 95 轮，样本太早，暂不据此评价学习效果。
- 该 run 的高速命令课程计数从新训练第 0 轮开始：`dash` 在第 2,000 轮启用，`spin_low` 第 3,000 轮启用，`spin_mid` 第 4,000 轮启用。因此从 1500 checkpoint 加载权重不等于保留原实验的课程进度。

## 项目脉络

### 1. XML、USD 与 articulation

Dogleg 的 MuJoCo 描述来自 `wheelbipe_ros2_sim2sim/src/resources/robot_descriptions/dogleg/mjcf/dogleg_floating_source.xml`。Isaac Lab 侧通过 USD 导入，并将并联机构的连接点作为 `PhysicsSphericalJoint` 闭链约束处理。Dogleg 运行时 articulation 有 28 个 body、27 个关节，另外找到 10 个闭链球形关节。

这 10 个约束是五个实体连接位置两侧各一对锚点：左右 calf-coupler、左右 rocker-ternary，以及 pitch rod 到 pitch link。不要把“碰撞体”和“闭链”当成同一层：碰撞几何处理机器人与地面/物体接触，闭链约束处理连杆之间的运动关系。

Dogleg 的腿机构和 V14 同属并联轮足构型，但 joint frame、坐标定义、尺寸和零位并不相同。V14 解析 IK 的长度/偏置不能直接照搬到 Dogleg；能启动、能训练也不能证明随机 reset IK 正确，因为之前 Dogleg 训练关闭了腿部 IK 随机 reset。

### 2. 碰撞体排查

碰撞模型曾经历简化、轮地接触检查和候选比较。把粗盒进一步拆成复合盒的候选仍有 13 个测试步触发非期望接触，没有转正。

当前正式的 contactfix overlay 恢复 base_link 和左右 front2_link 的原凸网格碰撞体，并关闭这三处粗盒；其他简化碰撞体和全部 10 个闭链约束保留。单环境落地、行驶与滑行回归中没有出现非期望接触惩罚，Isaac 中也验证了显示模型可见。这个结果支持 contactfix 作为当前训练资产，但不等于 PhysX 和 MuJoCo 的所有物理参数已经完全一致。

在 Isaac Sim 5.1 中显示碰撞形状的入口是 `Window > Physics > Debug`，再启用 `Simulation Debug Visualization > Enabled > Collision Shapes`。先前给的 Script Editor 同步调用曾触发原生 mutex 断言，应使用 Physics Debug 面板。

### 3. V14 IK 与 Dogleg 专用 IK

V14 Flat-v0 开启 `use_leg_random_start`，会随机采样腿长/腿角、使用 V14 的解析 IK 生成关节角；预设地面姿态也参与一部分 reset。V14 的标准腿长范围约 `0.13–0.32 m`、角度约 `−90°–135°`，预设姿态还会使实际角度达到约 `−115.4°`。

之前将 V14 解析 IK 原样作用于 Dogleg 的 `0.20 m、0°` 姿态，实测最大闭链残差约 `72.8 mm`。这证实原参数不能直接用，不代表同构机构无法解析求解。随后把 Dogleg 自身 MJCF 的关节和闭链锚点作为数值模型，离线求解每个腿长/角目标，再把关节解保存成 47 x 101 的二维查找表。训练运行时使用 Torch 双线性插值，避免每个环境 reset 都进行 SciPy/MuJoCo 迭代。

已在 Isaac contactfix USD 上进行验证：

- 1,000 组随机目标，腿长 `0.13–0.36 m`、腿角 `−120°–135°`，闭链残差中位数 `0.0062 mm`、95 分位 `0.0150 mm`、最大 `0.0188 mm`。
- 同批样本的轮心目标位置误差最大 `0.0151 mm`。
- 组合边界姿态（腿长 `0.13/0.36 m`，腿角 `−120°/135°`）最大闭链残差约 `0.001 mm`。

直接拟合 V14 解析公式的 Dogleg 长度和偏置，随机姿态闭链残差仍最高约 `17.4 mm`，所以当前不把这条近似解析式接入 reset。查找表是已验证的适配方案，不是 MuJoCo 每个仿真步都在执行 IK。

### 4. 训练阶段和关键实验

V14 Flat 的命令本身包含约 10% 零速环境；它不要求先做一个独立的纯定点阶段才能进入 Rough。V14 Rough-v1 则把平地、台阶、斜坡等混合地形、地形扫描与对应命令覆盖结合起来。Flat 奖励没有必须达到 100 的门槛，进入 Rough 应看基本移动、倒地率和策略稳定性，而不是要求总奖励“吃满”。

| 实验 | 配置与继承 | 可复盘的结论 |
| --- | --- | --- |
| `2026-11-23_23-14-11_dogleg_neutral_resume` | 16,384 环境；旧 Dogleg USD；10% 零速；`stand_still_lin_vel=-1`；从 `neutral_scratch/model_0.pt` 恢复优化器和迭代状态。 | 训练到 2,725 左右，平均 episode length 接近 999，平地姿态/高度稳定。后期总奖励约 79–80，线速度跟踪仍缓慢改善；总奖励变平不等于所有能力停止增长。高速 `dash` 第 2,000 轮开始，旋转阶段第 3,000/4,000 轮开始。 |
| `2026-11-24_09-03-36_stand_from_model2000_16384` | 16,384 环境；全零速度命令；从上一 run 的 model_2000 只载入策略权重，优化器和计数重置；最多 3,000 轮。 | 总奖励约 93.6，但这是纯零速专门任务，不能与混合速度统一策略的总奖励横向比较。 |
| `2026-11-24_11-32-26_stand_linvel_penalty2_from1000` | 从 stand run 的 model_1000 载入权重；纯零速；漂移惩罚系数从 −1 改成 −2；最多 2,000 轮。 | 惩罚系数改变会直接缩放 `Episode/Reward/stand_still_lin_vel`。比较时应按权重绝对值归一化，不能只看加权曲线是否更负。 |
| `2026-11-26_01-54-48_dogleg_contactfix_unified_scratch_20000_highspeed6000_zero10` | contactfix USD；20,000 环境；零速 10%；从随机初始化开始；当时腿姿态随机 IK 仍关闭。 | 约 1,000 轮时平均奖励约 80、线速度跟踪约 0.77、高度紧跟踪约 0.90、episode length 接近 999。其后 checkpoint `model_1500_contactfix.pt` 成为本次权重来源。 |
| `2026-11-26_06-44-42_dogleg_contactfix_from1500_weights_20000_zero10_ik` | 从上行的 model_1500 只继承策略权重；优化器和迭代计数清零；启用 Dogleg 随机 reset IK；20,000 环境、零速 10%。 | 当前 run 仍在进行。核对时约第 95 轮，属于 reset 分布变化后的早期阶段，不应据此判断是否收敛或与旧 run 总奖励直接比较。 |

注意：`DOGLEG_TRAINING_GUIDE.md` 保留的是较早的 512 环境 GUI、零速 0%、从随机初始化的启动快照，和当前启动脚本不一致。启动前以 `scripts/start_dogleg_unified_training.sh` 及新 run 的 `params/env.yaml`、`params/agent.yaml` 为准。

## 日志怎么读

- `Train/mean_episode_length` 接近最大长度 999，表示较少提前倒地/终止；它不证明速度跟踪或定点漂移优秀。
- `Episode/Reward/track_lin_vel_xy` 是线速度误差经指数奖励函数变换后的加权 episode 贡献，不是直接测得的 m/s。命令分布变化（如高速课程开启）会改变该曲线的统计难度。
- `Episode/Reward/track_height_exp_tight` 看高度命令跟踪；`flat_orientation_*` 看姿态稳定性；`Episode/Reset/terminate` 和 `time_out` 用于区分提前终止与正常超时。
- `Episode/Reward/stand_still_lin_vel` 在 `|target vx| < 0.1 m/s` 时惩罚水平机体速度绝对值之和。它不直接测量累计位置漂移或偏航；曲线是按 episode 归一化且已乘奖励权重的。跨权重比较先除以权重绝对值，混合零速和纯零速 run 仍不可直接比较。
- PPO 中 `Loss/entropy` 是策略分布熵相关指标，不要求递增；`Policy/mean_noise_std` 更直接反映动作探索尺度。`Loss/value_function` 是价值网络拟合误差，不是总奖励。
- PPO 的 `learning_rate` 使用 adaptive/KL 调整。checkpoint 恢复优化器会继续旧 Adam 动量和 optimizer 状态；只加载策略权重则从配置学习率重新开始。

## 常见误区与处理

- **迭代数不等于并行环境数。** `iteration 1500` 是 PPO 更新次数；每轮样本量约为 `num_envs x num_steps_per_env`。20,000 环境、每环境 24 步时，一轮是 480,000 个 transition。
- **`--headless` 只关闭图形窗口，不等于 CPU 仿真。** 训练仍可用 `--device=cuda:0`；GPU PhysX 之外，环境管理、传感器、日志等仍会占 CPU。
- **改脚本或 USD 不会改正在运行进程里的配置。** 进程启动时会加载配置和资产；之后修改文件需要启动新进程才能生效。每次 run 的 `params/env.yaml`、`params/agent.yaml` 才是当次实际配置证据。
- **并行环境多会增加 PhysX 接触缓存需求。** 20,000 环境已作为可运行配置，并把 `gpu_max_rigid_patch_count` 设为 `2**22`；24,000 曾导致资源崩溃。不要只看 GPU 利用率就推断并行数可以无限增加。
- **不要删整个 TensorBoard run 目录来隐藏曲线。** 单删该 run 的 `events.out.tfevents.*` 就能从 TensorBoard 移除曲线，同时保留 checkpoint 和参数快照。本次 `2026-11-26_06-26-34_dogleg_contactfix_from1500_to6000_zero10_ik` 就只删了 event 文件。
- **checkpoint 名称和继承语义要分开确认。** 普通 `--checkpoint` 只加载兼容的策略权重、计数归零、Adam 新建；`--resume_training` 恢复权重、优化器和计数；当前训练入口还支持 `--resume_iteration_only`，用于加载权重和迭代计数、重置优化器。checkpoint 文件名本身不能说明是哪一种，检查 run 的 `params/agent.yaml` 和 `launch_checkpoint`。
- **ROS topic 存在只证明通信链路存在。** `/rl_network_input` 和 `/rl_network_output` 有数据，不代表关节顺序、速度符号、动作缩放、观测归一化或 MuJoCo 执行器一定与训练一致；需要固定命令逐项记录输入、输出和 joint state。
- **站立惩罚曲线不是累计漂移量。** `stand_still_lin_vel` 统计零速命令时的水平速度惩罚；它不直接告诉你机器人几秒内移动了多少，也不测偏航累计。部署端还要观测世界坐标位置和 yaw 随时间的变化。

早期有几次结论曾把“训练中能动”“能打开 sim”“topic 有数据”当作模型或物理映射正确的充分证据。复盘时应将这些分别视作必要的冒烟测试，完整判断仍要依赖闭链残差、碰撞回归和跨仿真固定命令对照。

## 训练/部署差异排查经验

图形界面里看起来稳定，而 MuJoCo sim2sim 有缓慢漂移或转向偏差时，不要先只改奖励。按层核查：

1. 策略输入/输出顺序、归一化、动作 scale 和左右轮/左右腿符号。
2. 关节轴方向、初始零位、目标角裁剪和执行器力矩/速度限制。
3. 左右轮心位置、轮地碰撞几何、摩擦和 base/leg 质量惯量。
4. 闭链约束的锚点、厚度/偏置、PhysX 与 MuJoCo 求解器差异。
5. 固定同一命令和初始姿态，记录命令、观测、策略输出、关节状态和轮速，再做单变量对照。

查看 ROS 话题曾确认 `rl_network_input`、`rl_network_output` 在发布，说明网络链路活着，但这不足以证明关节/轮子观测映射完全一致。部署端仍应使用固定命令的轨迹和 joint state 做逐项对照。MuJoCo `solref` 属于闭链约束的求解/柔顺参数，不是 PPO 权重；改它会改变部署物理行为，需单独验证 sim2sim。

## 常用操作

当前启动脚本（从 1500 checkpoint 只继承策略权重，6000 个新迭代）：

```bash
cd /root/gpufree-data/wheeled-legged_RL
./scripts/start_dogleg_unified_training.sh
```

TensorBoard：

```bash
cd /root/gpufree-data/wheeled-legged_RL
/workspace/isaaclab/_isaac_sim/python.sh -m tensorboard.main \
  --logdir="$PWD/logs/rsl_rl/dogleg_flat_direct" --host=0.0.0.0 --port=6006
```

验证当前 USD 中 Dogleg IK 的随机范围：

```bash
cd /root/gpufree-data/wheeled-legged_RL
export PYTHONPATH="$PWD/source/agent_world:$PWD/source/agent_tasks:$PWD/source/agent_rl:$PWD"
TERM=xterm-256color /workspace/isaaclab/isaaclab.sh -p scripts/verify_dogleg_ik.py \
  --task=Robotics-Dogleg-Unified-Flat-v0 --device=cpu --num_envs=1 --headless --samples=1000
```

IK 查找表的源生成器依赖 MuJoCo、NumPy、SciPy，并读取部署仓库中的 `dogleg_floating_source.xml`；训练运行时只加载已生成的 `dogleg_ik_lut.npz`，不依赖 MuJoCo/SciPy。

## 后续建议

- 先让当前带随机 reset IK 的 run 运行数百轮，检查 episode length、termination、速度/高度跟踪和 reset 后接触稳定性；不要拿首 100 轮总奖励直接和固定中性姿态 run 比。
- 在确认随机 reset 没有显著恶化平地策略后，再用已训练 checkpoint 进入 Dogleg Rough；当前 Dogleg task 仍是 Flat，不能学会上台阶。
- 如果要保留总训练轮次课程进度，区分三种启动语义：只加载策略（优化器/计数归零）、仅恢复计数、恢复优化器和计数。当前启动脚本选择第一种；`dash` 因而从新计数第 2,000 轮开启。
- 每次资产、reset 逻辑或动作/观测定义变化，都记录 run 的资产路径与 git diff。对比结果时固定环境数、随机化、命令分布和训练初始化，只改变一个因素。

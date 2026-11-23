# Wheelbipe Rough Tasks

只看 rough 地形时，直接用下面这些 `task` 即可。

## 直接查看地形/机器人

使用脚本：

```bash
python scripts/view_robot.py --task <TASK_ID>
```

`view_robot.py` 现在会自动导入 `agent_tasks`，可以直接识别下面这些自定义 task。

## 推荐 task

### V13

- 普通 rough: `Robotics-Wheelbipe-V13-Rough-v0`
- 飞坡 rough: `Robotics-Wheelbipe-V13-Fly-Rough-v0`

### V14

- 普通 flat: `Robotics-Wheelbipe-V14-Flat-v0`
- 普通 rough: `Robotics-Wheelbipe-V14-Rough-v0`
- flat play: `Robotics-Wheelbipe-V14-Flat-Play-v0`
- rough play: `Robotics-Wheelbipe-V14-Rough-Play-v0`

### Wheelbipe25 V3

- 普通 rough: `Robotics-Wheelbipe25-V3-Rough-v0`
- 扩展 rough: `Robotics-Wheelbipe25-V3-Ext-Rough-v0`
- 飞坡 rough: `Robotics-Wheelbipe25-V3-Fly-Rough-v0`
- 飞坡扩展 rough: `Robotics-Wheelbipe25-V3-Fly-Ext-Rough-v0`
- Jump rough: `Robotics-Wheelbipe25-V3-Jump-Rough-v0`
- Jump 扩展 rough: `Robotics-Wheelbipe25-V3-Jump-Ext-Rough-v0`
- Serial rough: `Robotics-Wheelbipe25-V3-Serial-Rough-v0`

### Converted dogleg

- Flat smoke test and PPO training: `Robotics-Dogleg-Flat-v0`
- Neutral joint pose is calibrated to the V14.2 zero-pose wheel vector; action targets are clipped to ±3.14 rad around that pose. The USD leg joints remain continuous (no hard position stops).

MuJoCo checkpoint smoke test (requires `mujoco`, `torch`, and `numpy` in the active Python environment):

```bash
python scripts/mujoco_dogleg_policy.py --duration 20 --vx 0 --wz 0
```

Use `--no-viewer` for headless execution or set `--vx 0.3` for a low-speed transfer test. The adapter uses the calibrated neutral pose and 35-D policy observation layout; MuJoCo contact/actuator dynamics are only an approximation of PhysX.

## 怎么选

- 只看传统 rough 地形：选普通 `Rough-v0`
- 要看独立飞坡任务：选 `Fly-Rough-v0`
- 要看 jump 任务：选 `Jump-Rough-v0`
- 要看串行任务：选 `Serial-Rough-v0`
- 名字里带 `Ext` 的版本，是扩展配置，不是 play

## 地形对应关系

- 普通 rough env 走 `RM_ROUGH_TERRAINS_CFG`
- 飞坡 rough env 走 `RM_FLY_TERRAINS_CFG`

## 最常用示例

```bash
python scripts/view_robot.py --task Robotics-Wheelbipe-V13-Rough-v0
python scripts/view_robot.py --task Robotics-Wheelbipe-V13-Fly-Rough-v0
python scripts/view_robot.py --task Robotics-Wheelbipe-V14-Rough-Play-v0
python scripts/view_robot.py --task Robotics-Wheelbipe25-V3-Rough-v0
python scripts/view_robot.py --task Robotics-Wheelbipe25-V3-Fly-Rough-v0
python scripts/view_robot.py --task Robotics-Dogleg-Flat-v0
```

## 维护说明

- V14 不再包含 `guide_*` 机构，新增 `gimbal_yaw_*` / `gimbal_pitch_*` 两级云台关节，并在任务内部固定到零位。

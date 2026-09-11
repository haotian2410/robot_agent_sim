# robot_agent_sim × robot-agent-control

本项目保留两个独立 package：`robot_agent_sim` 负责“做什么”（自然语言、Route A/Route B、场景和 SkillPlan），`robot-agent-control` 负责“怎么运动”（命令转换、IK、夹爪和 MuJoCo runtime）。sim 只依赖 control 的公开契约；control 不依赖 sim 的内部模块。

## 环境

先保留原环境，再使用克隆环境做整合：

```bash
conda create -n robot_agent_integ --clone robot_agent
conda activate robot_agent_integ
python -m pip install -e /home/cscvlab/lht/robot-agent-control
python -m pip install -e /home/cscvlab/lht/robot_agent_sim
```

第一阶段支持 Python 3.11–3.12、NumPy 2.x、MuJoCo 3.3–3.x；当前执行后端只支持 UR5e。Panda 仍可规划，但进入控制阶段会明确拒绝。

## 四步接口

```bash
robot-agent-sim plan "把红色方块放进蓝色盒子" --robot ur5e --output-dir var/task01
robot-agent-sim compile var/task01
robot-agent-sim execute var/task01 --viewer-mode headless
robot-agent-sim run "把红色方块放进蓝色盒子" --robot ur5e --viewer-mode auto
```

`execute` 不调用 Qwen/VLM，只消费 `execution_bundle.json`，因此可以离线重放。`headless` 用 EGL；`auto` 连续执行并保留窗口；`step` 在每个 runtime step 后等待 Space/Enter。规划和执行通过子进程隔离，避免 EGL renderer 与 GLFW viewer 在同一进程冲突。

## Route A / Route B

* Route A（不传 `--scene`）自动生成最终 `scene.xml` 和简单物体/盒子/按钮的 `interaction_registry.json`。普通可抓物体使用 free joint，以便 control 在同一 runtime 中持久携带；柜门/抽屉等机构不会由几何猜测。
* Route B（传 `--scene`）优先使用同目录的 `<scene-stem>.interactions.json`，也可用 `--interaction-registry` 指定 sidecar。已知 sidecar 会校验 scene SHA-256 和 MuJoCo 的 body/site/joint/geom；没有执行元数据时仍可完成 VLM+IoU 规划，但 compile 会安全地返回 `execution_metadata_missing`。

每个任务目录会保存 `task_intent.json`、`scene_registry.json`、`grounded_task.json`、`skill_plan.json`、`interaction_registry.json`、`commands.json`、`execution_bundle.json`、`execution_report.json` 和 `skill_trace.jsonl`。`SKILL.md` 是面向 LLM/开发者的说明；运行时使用 Python skill catalog 和 versioned command contract，不在每次执行时解析 Markdown。

## Qwen

只有 `plan` / `run --provider qwen` 才要求 `QWEN_BASE_URL` 与 `QWEN_MODEL`（API key 可选，使用 `QWEN_API_KEY`）；`compile` 和 `execute` 完全不依赖 Qwen。Qwen 输出仍限制在结构化语义对象、操作和技能计划，物理 XYZ、关节角和轨迹参数由 registry/profile/compiler 提供。

## 分层验收

```bash
python -m pytest /home/cscvlab/lht/robot-agent-control/tests
python -m pytest /home/cscvlab/lht/robot_agent_sim/tests
robot-agent-control-demo --commands /home/cscvlab/lht/robot-agent-control/demo/skill_command/config_001.commands.json --viewer-mode headless
```

原 `config_001` golden fixture 和 Route A/Route B 规划入口均保留；完整执行只有一个持久化 `MjModel`/`MjData`，失败采用 fail-fast，并在 report/trace 中追溯到 SkillPlan step。

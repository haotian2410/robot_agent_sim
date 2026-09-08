# robot-agent-sim

`robot-agent-sim` 是一个只做任务理解、场景语义对齐和技能规划的 Python 项目。它接收中文或英文自然语言，输出 `TaskIntent`、对象绑定结果和 `SkillPlan`。当前命令不会让 Panda 或 UR5e 执行动作，也不做 IK、轨迹规划、碰撞规划或真实机器人控制。

主链路固定为：

```text
UserInstruction → TaskIntent → Scene Construction / Visual Grounding
→ GroundedTask → SkillPlan → Atomic Skill Sequence
```

Python 显式控制阶段和调用次数；模型不能自主 tool calling、循环或增加调用次数。MuJoCo 只负责场景、RGB 和 instance identity，VLM 只返回二维框，IoU 才负责把二维框绑定到仿真实例。

## 目录和核心 Schema

```text
src/robot_agent_sim/
├── contracts/       # TaskIntent / GroundedTask / SkillPlan（Pydantic）
├── models/          # Fake、Qwen Task Understanding / Vision / Planner
├── assets/          # AssetRegistry 与 AssetResolver
├── scene/           # 方向、SceneRegistry、确定性 SceneComposer
├── grounding/      # segmentation 与 [ymin,xmin,ymax,xmax] IoU
├── backends/mujoco/ # Panda/UR5e XML、渲染器、实例索引
├── pipeline/        # 两条 Route 的统一 PipelineEngine
└── cli.py           # robot-agent-sim plan
```

核心对象分别表示：

- `TaskIntent`：任务状态、任务类型、实体、顺序 operations 和空间关系；不含 XYZ、object_id 或 Skill。
- `GroundedTask`：每个语义实体绑定到唯一 `object_id`，记录 `asset_scene_binding` 或 `vlm_iou` 方法。
- `SkillPlan`：每个 `SkillStep` 含 operation、注册的 `skill_name`、对象引用和由 Python 补出的依赖关系。

三个模型 Prompt 位于 `src/robot_agent_sim/models/prompts.py`：Task Understanding 只抽取语义，Vision Grounding 只输出 0–1000 的二维 bbox，Skill Planning 只从固定 Atomic Skill Catalog 生成高层步骤。

项目是独立的，位于：

```text
/home/cscvlab/lht/robot_agent_sim
```

它自带复制到本项目内的 Panda/UR5e MuJoCo 资产，因此安装和运行时不需要从旧 `robot_agent` 包导入代码。

## 安装

终端刚打开时通常位于 `/home/cscvlab`，那里没有 `pyproject.toml`。先进入项目目录，再安装：

```bash
conda activate robot_agent
cd /home/cscvlab/lht/robot_agent_sim
python -m pip install -e '.[dev]'
```

`-e` 表示 editable install：代码仍保存在当前目录，但 `robot-agent-sim` 命令会注册到当前 Conda 环境。安装完成后可以在任意目录运行该命令；如果不想安装，也可以在项目目录用 `PYTHONPATH=src python -m robot_agent_sim.cli ...`。

## 正确的 CLI 形式

`plan` 是子命令，任务文本放在它后面：

```bash
robot-agent-sim plan \
  "把左边的红色方块放进右边蓝色盒子" \
  --robot panda \
  --seed 7
```

这里的 `7` 是随机布局种子。它不是场景编号、模型编号或机器人编号。省略时默认使用 `0`；相同任务、相同机器人和相同 seed 会得到相同的自动布置位置，换成其他整数会得到另一组可复现位置。例如：

```bash
robot-agent-sim plan "抓取红方块" --seed 0
robot-agent-sim plan "抓取红方块" --seed 7
robot-agent-sim plan "抓取红方块" --seed 42
```

如果任务已经明确给出左、右、前或后等方向，方向会优先决定位置，seed 只影响没有明确方位的对象。当前支持：

```text
left/right/front/back/up/down
左/右/前/后/上/下
东=右，西=左，南=后，北=前
```

“东北、左前方、斜上方”等组合方向会返回 `direction_clarification_required`，不会自动近似。

自动场景统一使用桌面中心坐标系：`+x=右、-x=左、+y=前、-y=后、+z=上、-z=下`。生成场景的 `scene_registry.json` 中，`position` 表示物体底面与桌面的接触位置；网格导入时会根据 OBJ bbox 自动把几何底部校正到该位置。它是语义场景真值，不是控制接口或机器人执行坐标。

## 参数说明

运行：

```bash
robot-agent-sim plan --help
```

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `instruction` | 必填 | 用户自然语言任务。含空格时使用引号。 |
| `--robot` | `panda` | 选择 `panda` 或 `ur5e`。Route A 用于场景构造；Route B 用于记录选定机器人。 |
| `--seed` | `0` | 自动生成场景的随机种子，任意整数。相同 seed 可复现。 |
| `--scene` | 不指定 | 已有 MuJoCo 场景 XML/MJCF 文件路径；指定后进入 Route B。 |
| `--output-dir` | `var` | 输出目录。相对当前工作目录；已有同名 artifact 会被覆盖。 |
| `--provider` | `fake` | `fake` 离线测试，或 `qwen` 使用 OpenAI-compatible 服务。 |
| `--planner` | `recipe` | `recipe` 确定性规划、`qwen` 实验规划、`auto` 自动选择。 |
| `--structured-output` | `json_schema` | Qwen 输出约束：`json_schema`、`json_object` 或 `off`。 |
| `--qwen-base-url` | 不指定 | Qwen API 根地址；也可用 `QWEN_BASE_URL`。 |
| `--qwen-model` | 不指定 | 服务端模型名；也可用 `QWEN_MODEL`。 |

## `--scene` 是否必须指定

不必须。

省略 `--scene` 时走 Route A：系统根据任务实体从本地 Asset Catalog 选择资产，自动生成桌面场景，并生成 `scene.xml`、RGB 和 instance segmentation：

```bash
robot-agent-sim plan \
  "把红色方块放进蓝色盒子" \
  --robot panda \
  --seed 7 \
  --output-dir var/pick-place
```

Route A 的目录是 `assets/models`，当前会扫描以下网格资产：

```text
apple, banana, baseball, rubiks_cube, sponge, spoon, sugar_box
```

每个目录的 `textured.obj` 是可渲染网格，`texture_map.png` 是可选纹理，`textured_coacd_*.stl` 是可选碰撞网格。`*.json` 元数据只用于别名和补充信息，不是运行入口；缺少 `cube`、`open box` 或 `button` 网格时，系统会使用受控 primitive 补齐。资产解析不到所需模型时返回 `asset_missing`，不会让模型直接指定 MuJoCo `object_id`。

指定 `--scene` 时走 Route B：系统加载你提供的 XML/MJCF，渲染 RGB 和 instance segmentation，再执行视觉检测步骤：

```bash
robot-agent-sim plan \
  "把左边红色物体放进蓝色容器" \
  --robot ur5e \
  --scene /absolute/path/to/scene.xml \
  --output-dir var/uploaded-scene
```

`--scene` 文件必须在执行命令时已经存在，并且扩展名是 `.xml` 或 `.mjcf`。当前命令不会下载或生成上传场景；文件不存在、扩展名错误、XML 不是 MuJoCo 根节点，或包含禁止的 `include/plugin` 结构时会直接报错。

## 内置 XML 场景

项目内置的 XML 主要用于机器人模型和 Route A 生成后的场景基础：

```text
assets/robots/panda/panda.xml
assets/robots/panda/panda_nohand.xml
assets/robots/panda/scene.xml

assets/robots/ur5e/scenes/scene_000.xml
assets/robots/ur5e/scenes/scene_001.xml
assets/robots/ur5e/scenes/scene_002.xml
assets/robots/ur5e/scenes/scene_003.xml
```

UR5e 的 `scene_000` 到 `scene_003` 都是带桌面和固定相机的示例场景；其中 `scene_000` 只包含基础工作台（没有任务物体），`scene_001`–`scene_003` 还包含可被自动发现的任务物体。它们可以直接作为 `--scene` 参数，例如：

| XML | 主要内容 | 适合的示例 |
|---|---|---|
| `scene_000.xml` | UR5e、Robotiq 夹爪、工作台、相机，无任务物体 | 检查基础机器人场景 |
| `scene_001.xml` | 绿色盒子、红球、蓝色柜体、抽屉方块 | 物体定位和容器类任务 |
| `scene_002.xml` | 玻璃杯、橙色球、屏幕、托盘 | 多类别物体定位任务 |
| `scene_003.xml` | 按钮底座 | 按按钮任务 |

```bash
robot-agent-sim plan \
  "按按钮" \
  --robot ur5e \
  --scene assets/robots/ur5e/scenes/scene_003.xml \
  --output-dir var/ur5e-button
```

Panda 的 `panda.xml` 是机器人模型文件，本身没有任务物体和完整任务桌面。省略 `--scene` 时，程序会复制它的 MuJoCo 定义，在输出目录生成带桌面、相机和任务物体的 `scene.xml`。因此日常使用不需要手动指定内置 XML。

上传场景中的机器人由 XML 自身决定。 `--robot ur5e` 是本次规划选择和记录的机器人，不会把一个 Panda XML 自动改写成 UR5e XML；建议让 `--robot` 与 XML 内容保持一致。

## 输出文件

例如 `--output-dir var/pick-place` 会生成：

```text
var/pick-place/
├── task_intent.json
├── scene_registry.json
├── grounded_task.json
├── visual_grounding.json # Route B；Route A 不生成
├── skill_plan.json
├── summary.json
├── model_usage.json
├── scene.xml              # Route A 生成场景时存在
├── rgb.png
├── segmentation.npy
├── instances.json
└── instance_index.json    # compatibility alias of instances.json
```

`skill_plan.json` 中每一步包含注册 Skill 和对象引用；中文描述由 Registry 确定性生成。例如 pick-and-place 的顺序是：

```text
locate → move → grasp → locate → move → release
```

当前固定 Skill Catalog 是：

```text
locate, search, move, grasp, release, press
```

## Provider 和模型调用次数

默认 CLI 使用 Fake Providers，所以不需要 GPU、联网或 Qwen 服务。当前 Python 接口也提供 OpenAI-compatible Qwen HTTP Provider，见 `src/robot_agent_sim/models/qwen_http.py`；接入真实服务时需要在 Python 中注入 `base_url`、`model` 和 `api_key`。

CLI 也可显式选择 Qwen；API key 只从环境变量读取，避免出现在 shell history：

```bash
export QWEN_BASE_URL=http://localhost:8000/v1
export QWEN_MODEL=Qwen3-VL
export QWEN_API_KEY=your-key-if-required

robot-agent-sim plan "抓取苹果" \
  --robot panda \
  --provider qwen \
  --output-dir var/qwen-apple
```

默认 Route B 也会执行完整的 `VisionGroundingProvider → IoU → object_id` 主流程。Fake Vision Provider 的默认框只是测试占位，通常会因为与真实框不重合而返回 `grounding_failed`；在 Python 中注入带有审计过的 bbox 的 Fake Provider，或接入 Qwen HTTP Provider，即可得到成功绑定。系统永远不会让 VLM 直接猜 MuJoCo `object_id`。

Route B 额外保存：

```text
visual_grounding.json   # 检测框、真值框、IoU、unmatched/ambiguous
```

如果 XML 同目录存在 `<scene-stem>.scene_registry.json`，它优先定义 `object_id/body_name/semantic_name`。没有 sidecar 时，程序自动发现 XML 中的顶层任务物体 body，并排除机器人、桌面、相机、灯光等基础设施。`scene_000.xml` 没有任务物体，直接作为 Route B 输入时应配 sidecar 或改用 `scene_001`–`scene_003`。

默认 Pipeline 固定统计调用次数：

```text
Route A + recipe：Task Understanding = 1 次
Route B + recipe：Task Understanding + Vision Grounding = 2 次
Route A + qwen：2 次；Route B + qwen：3 次
```

系统没有 Agent Loop，也不会根据模型输出自动增加调用轮数。真实 Qwen Provider 是 OpenAI-compatible HTTP：

```python
from robot_agent_sim.models.qwen_http import QwenHTTPProvider
from robot_agent_sim.pipeline.engine import PipelineEngine

provider = QwenHTTPProvider(
    base_url="http://localhost:8000/v1",
    model="Qwen3-VL",
    api_key="",
)
engine = PipelineEngine(understanding=provider, vision=provider, planner=provider)
result = engine.plan("把左边的红色方块放进右边蓝色盒子", robot="panda", seed=7, planner="recipe")
```

Provider 不重试请求；每次固定阶段调用都会记录在 `provider.calls`，Pipeline 结果中的 `model_call_count` 是实际已发起的阶段调用数。
每个阶段使用独立的 completion 上限，结果中的 `model_usage` 和 `model_usage.json` 记录 input/output/total token；预算超限会直接返回 `model_call_budget_exceeded`，不会隐式重试。

`recipe` 模式覆盖 locate/search/move/grasp/release/press/pick_and_place 等标准任务，完全不调用 Skill Planner。`qwen` 模式保留用于复杂任务实验；`auto` 在 RecipePlanner 不支持时才回退到 Qwen，并仍受 Route A ≤2、Route B ≤3 的硬预算限制。

## 测试

```bash
cd /home/cscvlab/lht/robot_agent_sim
python -m pytest -q
```

当前测试覆盖 Route A 规划、mesh/primitive 资产、seed 可复现、方向和 unsupported task、选择关系、混合任务、Route B IoU 成功/失败、sidecar/自动发现、技能顺序校验、Qwen 请求协议和 CLI 参数解析。运行测试时以终端显示的实际数量为准：

```bash
python -m pytest -q
```

# UR5e 组合模型来源

本目录的 UR5e、Robotiq 2F-85 和 Intel RealSense D435i 网格来自 Google DeepMind
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，固定上游 commit：

```text
8161bba264d7fa7c99ca301e91e7fb44737676ad
```

2026-09-06 审计时，以下目录中的每个网格文件均与该 commit 逐文件 SHA-256 一致：

- `assets/ur5e/` 对应 `universal_robots_ur5e/assets/`；
- `assets/robotiq_2f85/` 对应 `robotiq_2f85/assets/`；
- `assets/overhead_camera_d435i/` 和 `assets/wrist_camera_d435i/` 均对应
  `realsense_d435i/assets/`。

场景 XML 是项目内用于测试的组合场景，不是 Menagerie 原始场景。原外部场景中的自定义按钮环
没有可验证来源，因此没有纳入可信资产；Scene 003 改用 MuJoCo `cylinder` 基本几何表示按钮底座。

第三方许可全文见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。RealSense D435i 使用
Apache License 2.0，完整许可证文本同时保存在仓库
`assets/robots/franka_emika_panda/LICENSE`。

# motionUse

五 IMU 实时人体姿态项目：

```text
SlimeVR -> SolarXR -> MobilePose -> SMPL 24 joints
        -> local WebSocket -> Unity SMPL / Humanoid Avatar
```

当前重点不是继续调整 Unity，而是定位自然站立时实时 IMU 输入与训练数据之间的固定旋转偏差。
完整文档按阅读顺序收录在 [docs/README.md](docs/README.md)。当前状态、已验证结论和下一步任务见
[项目交接](docs/22_项目交接.md)。

## 核心目录

| 目录 | 职责 |
| --- | --- |
| `mobileposer/` | Python 模型、训练、实时推理、录制回放和 Web 诊断 |
| `IMUTrack-for-Spine/` | Unity 项目，包含官方 SMPL 直驱与 Humanoid 重定向 |
| `SlimeVR-Server/` | SlimeVR Server 源码和桌面 UI 构建 |
| `solarxr_protocol/` | Python SolarXR 协议生成代码 |
| `tests/` | Python 自动化测试 |
| `scripts/` | 本地环境和受限资产安装脚本 |

## 快速启动

```powershell
conda activate mobileposer
python -m mobileposer.realtime.run `
  --checkpoint "results\ours_multi_finetune\layouts\wrists_shanks_waist\model_finetuned.pth" `
  --calibration-frames 150 `
  --output-fps 30 `
  --include-contact `
  --contact-interval 60 `
  --web
```

网页地址：`http://127.0.0.1:8765/`

Unity 场景：`IMUTrack-for-Spine/Assets/Scenes/AvatarScene.unity`

## 本地资产

SMPL 模型、数据集、checkpoint、录制和构建依赖不会上传 Git。取得合法 SMPL v1.0.0 下载后运行：

```powershell
.\scripts\install_smpl_assets.ps1
```

SMPL 资源须遵守其研究用途许可证。

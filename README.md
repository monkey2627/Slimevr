# 五 IMU 实时人体姿态定位

本项目当前唯一主线是：

    SlimeVR -> SolarXR -> MobilePose -> SMPL 24 关节旋转
             -> 本地 WebSocket -> Unity Humanoid Avatar

目标不是把五个 IMU 直接当作 Avatar 骨骼，而是建立一条可记录、可回放、可比较的实时姿态链路，用来判断误差来自 SolarXR 输入、MobilePose 模型还是 Unity 重定向。

当前布局为 wrists_shanks_waist：

    LEFT_LOWER_ARM  -> lw
    RIGHT_LOWER_ARM -> rw
    LEFT_FOOT       -> ls
    RIGHT_FOOT      -> rs
    WAIST           -> waist

## 环境

使用已有的 Conda 环境：

    conda activate mobileposer

确认环境：

    python -c "import torch; print(torch.__version__)"

## 实时启动

    python -m mobileposer.cli.realtime.run --checkpoint "D:\dyh\MotionRe\base_mobileposer\results\ours_multi_finetune\layouts\wrists_shanks_waist\model_finetuned.pth" --config configs/realtime/default.yaml --record recordings\session.jsonl

兼容入口仍然可用：

    python -m mobileposer.realtime.run ...

Unity 中只保留一个 MobilePoserPoseSource，初始基线保持：

    Use Tracker Anchors = false
    Use Relative Foot Grounding = false
    Enforce Joint Limits = false
    Guard Knee Hyperextension = false

## 录制、回放和比较

    python -m mobileposer.cli.replay.recording recordings\session.jsonl --report recordings\session.replay.json

    python -m mobileposer.cli.analysis.compare_inputs recordings\session.jsonl results\ours_multi_finetune\captures\<capture>\inputs\wrists_shanks_waist\ours1.pt

详细流程见 docs/ 目录。

## 目录约定

    mobileposer/realtime/  当前 SolarXR、校准、推理和协议实现
    mobileposer/core/      稳定的公共常量和运行时契约
    mobileposer/cli/       面向用户的统一命令入口
    configs/               实时、训练和评估配置
    docs/                  当前中文说明
    tests/                 可自动执行的测试
    data/                  输入数据
    checkpoints/           基础 checkpoint
    results/               训练和评估结果
    archive/               非当前主线的历史实验

训练、camera fusion、Android 导出和旧实验不是当前实时主线，只有在对应文档明确引用时才使用。
# Slimevr

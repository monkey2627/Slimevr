# SlimeVR 校准流程源码审计

本文基于本仓库中的 `SlimeVR-Server` 源码整理。这里的“校准”不是一个算法，而是几类不同操作：姿态参考重置、安装方向重置、用户身高/地面测量，以及 AutoBone 骨骼比例优化。硬件 IMU 的加速度计/陀螺仪 bias 标定不在本文所审计的 Server 代码中。

## 1. 总览

```text
Tracker 原始姿态
  -> mountingOrientation
  -> gyroFix / attachmentFix / mountRotFix / tposeDownFix / yawFix
  -> HumanSkeleton、IK、LegTweaks
  -> 用户身高尺度或 AutoBone 骨骼长度
  -> 输出人体姿态
```

每帧姿态修正的实际顺序见 [`TrackerResetsHandler.kt:203-223`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:203)：先应用手动安装方向、全重置产生的修正，再应用自动安装方向、T-Pose 修正、Yaw 修正和约束修正。

## 2. 全重置（Full Reset）

入口是 [`HumanSkeleton.kt:1550-1580`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/processor/skeleton/HumanSkeleton.kt:1550)，具体计算在 [`TrackerResetsHandler.kt:256-340`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:256)。Server 先重置 HMD，再用 HMD 的当前旋转作为其他 Tracker 的参考。

全重置的语义是：把当前 Tracker 姿态解释为 `(Pitch=0, Yaw=参考方向, Roll=0)`，而不是修改传感器内部状态。它为后续输出保存以下局部修正量：

- `gyroFix`：消除当前水平朝向；实现是当前 Yaw 四元数的逆，见 [`TrackerResetsHandler.kt:470`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:470)。
- `attachmentFix`：消除当前佩戴产生的 Pitch/Roll；普通 Tracker 使用 `(gyroFix * rotation).inverse()`，见 [`TrackerResetsHandler.kt:472`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:304)。
- `yawFix`：将校正后的 Tracker 水平朝向对齐 HMD/参考旋转，见 [`TrackerResetsHandler.kt:327-330`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:327)。
- `tposeDownFix`：在手臂采用 T-Pose 模式时加入左右 ±90 度转换，见 [`TrackerResetsHandler.kt:272-279`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:272)。

全重置完成后会清除 `needReset`、重置滤波器参考四元数，并重置 Stay Aligned/相关骨骼缓存，见 [`TrackerResetsHandler.kt:335-348`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:335)。因此它是“服务器输出参考重设”，不是 IMU bias 校准。

## 3. Yaw 重置

入口是 [`HumanSkeleton.kt:1582-1604`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/processor/skeleton/HumanSkeleton.kt:1582)，具体逻辑在 [`TrackerResetsHandler.kt:350-394`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:350)。

Yaw 重置只重算 `yawFix`，不重新计算 `gyroFix` 或 `attachmentFix`：

```text
yawFix = fixYaw(rawRotation * mountingOrientation, reference)
```

如果配置了 `yawResetSmoothTime`，旧、新 Yaw 修正之间会插值过渡，避免输出瞬间跳变，见 [`TrackerResetsHandler.kt:372-386`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:372)。

实现中的 `getYawQuaternion()` 从 YZX Euler 角直接提取 Yaw；源码明确注明这种方法在 Tracker 带有 Roll、尤其向前指向时可能受 Roll 影响，见 [`TrackerResetsHandler.kt:482-488`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:482)。所以这里不是严格的 swing-twist 分解。

## 4. 安装方向重置（Mounting Reset）

入口和限制检查在 [`HumanSkeleton.kt:1606-1645`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/processor/skeleton/HumanSkeleton.kt:1606)。如果存在 Tracker 的 `needReset` 状态，安装重置会被拒绝，要求先做全重置。

计算过程见 [`TrackerResetsHandler.kt:400-456`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:400)：

1. 取得已做基本参考修正的当前旋转；
2. 去掉参考旋转的水平朝向；
3. 用旋转后的世界向上向量计算 `atan2(x, z)` 得到安装 Yaw；
4. 根据左右手臂、手指、前臂 BACK/FORWARD、腿部等部位添加 ±90 度或 180 度补偿；
5. 保存为 `mountRotFix`，可选地持久化安装方向。

它解决的是“Tracker 绑在身体上时传感器朝向不一致”，不是改变骨骼长度，也不是重新估计陀螺仪偏置。

## 5. 用户高度/地面校准

实现为 [`UserHeightCalibration.kt`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/processor/skeleton/UserHeightCalibration.kt)。可执行条件是：至少有一个可用的手部位置 Tracker，以及一个可用的头部位置 Tracker，见 [`UserHeightCalibration.kt:124-145`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/processor/skeleton/UserHeightCalibration.kt:124)。

### 5.1 地面阶段

程序选取 Y 最低的手部 Tracker。它必须位于 `0.10 m` 以内、朝向地面（允许 45 度误差），并且位置稳定。位置样本最多 100 个，标准差阈值为 `0.005 m`，稳定持续约 300 ms 后记录地面最低 Y，见 [`UserHeightCalibration.kt:178-215`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/processor/skeleton/UserHeightCalibration.kt:178)。

### 5.2 站立阶段

头部相对地面高度为：

```text
relativeY = HMD.position.y - currentFloorLevel
```

要求 HMD 至少抬高 `1.2 m`，头部向上向量与世界向上方向夹角不超过 15 度。头部位置使用最多 100 个样本，标准差阈值为 `0.003 m`，稳定持续约 600 ms 后完成，见 [`UserHeightCalibration.kt:218-249`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/processor/skeleton/UserHeightCalibration.kt:218)。代码还将结果限制在 `1.2-1.936 m`；超出范围返回 TOO_SMALL/TOO_HIGH，见文件后部的状态判断。

### 5.3 应用结果

完成后写入 `skeleton.hmdHeight = currentHeight`、`floorHeight = 0`，随后重置骨骼 offsets 并保存配置，见 [`UserHeightCalibration.kt:150-159`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/processor/skeleton/UserHeightCalibration.kt:150)。因此这里测量的是 HMD/眼高参考和地面，不应简单描述成“直接测得完整人体身高”。骨骼的配置高度则是若干 `HEIGHT_OFFSETS` 之和，见 [`SkeletonConfigManager.kt:112-118`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/processor/config/SkeletonConfigManager.kt:112)。

## 6. AutoBone 骨骼比例优化

AutoBone 处理动作录制 `PoseFrames`，入口是 [`AutoBone.kt:214-295`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/autobone/AutoBone.kt:214)。可调整参数包括头、颈、胸、腰、髋、髋宽、大腿和小腿等 offset，见 [`AutoBone.kt:32-45`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/autobone/AutoBone.kt:32)。

流程如下：

1. 选择目标 HMD 高度：优先使用骨骼配置，否则使用录制中的最大 HMD 高度；
2. 归一化两个比较姿态的骨骼和 offsets，并按目标高度缩放；
3. 对每个 epoch 计算误差，试探每个 offset 增大或减小后的结果；
4. 仅保留能降低误差的更新；
5. 最后将归一化 offsets 按估计高度恢复尺度。

误差由 `getErrorDeriv()` 按配置权重累加，见 [`AutoBone.kt:526-560`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/autobone/AutoBone.kt:526)。其中 `SlideError` 衡量两帧之间脚部位置变化，见 [`SlideError.kt:17-52`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/autobone/errors/SlideError.kt:17)。人体比例误差将头、颈、胸、腰、髋、腿等 offset 与比例限制比较，见 [`BodyProportionError.kt:15-79`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/autobone/errors/BodyProportionError.kt:15)。

需要区分“代码支持的误差项”和“默认配置权重”：当前 [`AutoBoneConfig.kt`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/config/AutoBoneConfig.kt) 默认 `slideErrorFactor=1.0`、`bodyProportionErrorFactor=0.05`，而 offset 滑移、脚高、绝对高度、位置等误差权重默认是 0。因此不能笼统地说每次 AutoBone 都同时优化所有误差。

当启用帧过滤时，程序会按帧平均误差移除异常帧；但默认 `useFrameFiltering=false`。更新过程不是解析求解，而是启发式的误差下降迭代，核心试探和接受逻辑见 [`AutoBone.kt:426-457`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/autobone/AutoBone.kt:426)。

## 7. 漂移补偿与边界

漂移补偿是在重置之间记录姿态变化，并按时间将平均漂移四元数逐渐施加到输出，见 [`TrackerResetsHandler.kt:241-253`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:241)。它是输出层的渐进修正，不等价于重新估计 IMU 的硬件 bias。

本审计没有把固件中的传感器校准、磁力计校准或 AHRS 融合算法归入 Server 校准流程；若需要完整硬件链路，应另行审计 SlimeVR Tracker 固件仓库。

## 8. 结论表

| 操作 | 主要输入 | 修改内容 | 是否改变骨骼长度 |
|---|---|---|---|
| 全重置 | 当前 Tracker 姿态、HMD 参考 | `gyroFix`、`attachmentFix`、`yawFix` 等 | 否 |
| Yaw 重置 | 当前水平朝向、HMD 参考 | 仅 `yawFix` | 否 |
| 安装重置 | 当前已校正姿态、身体部位 | `mountRotFix` | 否 |
| 用户高度校准 | 手柄地面位置、HMD 位置和姿态 | `hmdHeight`、`floorHeight` | 间接重置 offsets |
| AutoBone | 多帧动作录制、目标 HMD 高度 | 骨骼 offsets | 是 |
| 漂移补偿 | 重置间姿态变化、时间 | 输出渐进补偿 | 否 |

## 9. 对 MobilePoser 输入的实际含义

本项目的 Python 实时端读取 SolarXR 的 `rotationReferenceAdjusted` 和 `linearAcceleration`，而不是 SlimeVR HumanSkeleton 计算出的脚踝/膝盖空间位置。

### 9.1 姿态字段

`rotationReferenceAdjusted` 是 SlimeVR 经过安装方向、Full Reset/Yaw Reset/Mounting Reset 等参考修正后的 Tracker 姿态。Python 端随后再做 Server 坐标系到 SMPL 坐标系的变换，以及前臂/上臂固定的左右 ±90 度变换，见 [`mobileposer/realtime/calibration.py:105-113`](../mobileposer/realtime/calibration.py:105)。

### 9.2 加速度字段

SlimeVR 的 `linearAcceleration` 通过 Tracker 的原始旋转 `_rotation` 将本地加速度转换到世界参考，见 [`Tracker.kt:454-460`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/Tracker.kt:454) 和 [`TrackerResetsHandler.kt:195-196`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:195)。它不是将加速度从“传感器实际位置”换算到绑定角色所代表的骨骼末端。

Python 启动阶段使用同一 Tracker 的 raw rotation 与 `rotationReferenceAdjusted` 做 heading 拟合，再将加速度变换到模型坐标系，见 [`mobileposer/realtime/calibration.py:124-181`](../mobileposer/realtime/calibration.py:124)。因此数据链是：

```text
SolarXR rotationReferenceAdjusted -> Python 姿态特征
SolarXR linearAcceleration       -> Python heading 对齐 -> 加速度特征
```

模型最终接收 `[acceleration / 30 | orientation]` 的 60 维输入，见 [`mobileposer/realtime/infer.py:58-67`](../mobileposer/realtime/infer.py:58)。

## 10. 绑定角色与物理安装位置

`BodyPart`/`TrackerPosition` 是 Tracker 的身份和路由标签，不是沿一根肢体的精确坐标。当前 GUI 英文文案将：

```text
LEFT_UPPER_LEG -> Left thigh
LEFT_LOWER_LEG -> Left ankle
LEFT_FOOT      -> Left foot
```

右侧同理。源码枚举定义在 [`TrackerPosition.kt:117-134`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerPosition.kt:117)，英文文案见 [`translation.ftl:45-57`](../SlimeVR-Server/gui/public/i18n/en/translation.ftl:45)。当前没有独立的“小腿上段”“小腿中段”“脚踝上方”角色。

把安装在膝盖下方的 Tracker 绑定为 GUI 的 `Left ankle`，不会让 SlimeVR 自动判断其离膝盖或脚踝的距离，也不会改变传感器实际输出的加速度。它会影响角色分类、默认 mounting orientation 和 SlimeVR 自身骨骼解算；Mounting Reset 只估计方向修正，不估计沿小腿的安装偏移，见 [`TrackerResetsHandler.kt:400-456`](../SlimeVR-Server/server/core/src/main/java/dev/slimevr/tracking/trackers/TrackerResetsHandler.kt:400)。

因此，Full Reset、Yaw Reset、Mounting Reset 和雪橇姿势校准都不会补偿不同安装点产生的杠杆臂效应。若传感器从膝下四分之一移动到脚踝，实际加速度会因

```text
a = a_origin + alpha x r + omega x (omega x r)
```

而改变；仅改变绑定名称或重新 Reset 不能消除该差异。

## 11. 当前 MobilePoser 布局的推荐映射

你当前的表面点训练布局应使用以下统一语义：

| 模型输入 | 训练表面点 | SlimeVR 绑定角色 | 说明 |
|---|---|---|---|
| `lw` | 左前臂表面，joint 18 | `LEFT_LOWER_ARM` | 不是手部中心 |
| `rw` | 右前臂表面，joint 19 | `RIGHT_LOWER_ARM` | 不是手部中心 |
| `ls` | 顶点 1176，joint 4，小腿上段 | GUI 的 `Left ankle` / `LEFT_LOWER_LEG` | 绑定名是角色名，物理位置仍应在训练点 |
| `rs` | 顶点 4662，joint 5，小腿上段 | GUI 的 `Right ankle` / `RIGHT_LOWER_LEG` | 同上 |
| `pelvis` | 顶点 3021，joint 0，骶骨表面 | `HIP` | 不应使用 `WAIST` 替代 |

Python 输入顺序必须固定为：

```text
[lw, rw, ls, rs, pelvis]
```

当前仓库中 `mobileposer/realtime/layout.py` 已将 `LEFT_LOWER_LEG/RIGHT_LOWER_LEG` 路由为 `ls/rs`，但第五个槽位仍命名为 `waist` 并映射 `WAIST`；真实预处理也使用 `WAIST`。若训练定义确实是 joint 0 的 pelvis 表面点，则需要把训练配置、真实微调预处理和实时 layout 一并改为 `pelvis -> HIP`，否则第五个输入槽位与训练定义不一致。

## 12. 输入正确性的判定标准

对 Python 模型而言，最重要的等式是：

```text
训练表面点
≈ 真实采集时的 IMU 物理安装点
≈ 实时 SolarXR 读取的同一个 Tracker
```

SlimeVR 的 Reset 负责方向参考；Python 的启动 heading calibration 负责把加速度和姿态放进模型坐标系；二者都不负责把脚踝传感器的信号变成小腿上段传感器的信号。若物理安装点不同，应重新生成对应 SMPL 网格表面点轨迹并重新训练/微调，或实现包含角速度、角加速度和刚体偏移的专门杠杆臂补偿。

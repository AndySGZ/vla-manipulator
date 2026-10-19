# SO101 π0.5 旧版 25k checkpoint：动作链路和失败诊断

本次只使用 `pi05_so101_lora`（`discrete_state_input=False`）和原先的 24,999 step checkpoint，没有使用后来训练的状态 token 版。闭环测试沿用 `replan-steps=5`、`max-steps=600`（20 Hz 仿真时间 30 秒）。挑选原 25k / max600 评估中每类一个失败 seed，共 8 条；这不是随机抽样的成功率估计。

## 测试结果

1. `openpi_ext/check_so101_roundtrip.py` 从正式 LeRobot 数据读取真实帧，依次运行 SO101 字段重排、输入变换、前五维 delta/夹爪绝对值、checkpoint 归一化与逆变换、动作恢复。跨数据集多个位置的样本通过，六维动作最大恢复误差为 0。测试不包含模型预测本身。
2. `simulation/mj_env/scripts/check_adapter_roundtrip.py` 验证了六维归一化电机坐标与 MuJoCo 弧度在范围内的往返，也确认超范围动作会被裁剪。
3. `simulation/mj_env/scripts/diagnose_openpi.py` 在真实闭环中记录 TCP 到目标中心距离、双指接触、抬升高度、其他物体接触、模型动作越界、关节跟踪误差、推理耗时；根据观测量标注失败阶段。它不从这些指标直接断言视觉识别、语言理解或权重是根因。

同一组 8 个 seed 第一次全部失败；第二次有 3 个成功、5 个失败。这表明固定场景 seed 没有固定模型采样轨迹，不能把两次差异单独归因于代码变化。第二次的可观察结果如下：

| 物体 | 第二次结果 | 最小 TCP—物体中心距离 | 双指接触步数 | 最大抬升 | 明显越界动作步数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 红杆 | 失败 | 30 mm | 9 | 8 mm | 5 |
| 黄杆 | 成功 | 21 mm | 35 | 63 mm | 0 |
| 蓝球 | 成功 | 23 mm | 19 | 63 mm | 5 |
| 绿球 | 失败 | 6 mm | 0 | 0 mm | 459 |
| 红方块 | 失败 | 29 mm | 14 | 7 mm | 48 |
| 蓝方块 | 失败 | 23 mm | 27 | 40 mm | 224 |
| 橙哑铃 | 失败 | 20 mm | 0 | 0 mm | 203 |
| 紫哑铃 | 成功 | 5 mm | 16 | 61 mm | 33 |

绿球与橙哑铃是“接近目标中心但没有双指抓取”；红杆、红方块、蓝方块是“出现过抓取接触但没有达到 6 cm 抬升”。所有失败轨迹的平均关节跟踪误差为 0.00046–0.01464 rad，并没有表现出执行器普遍跟不上位置目标。部分失败轨迹的模型输出持续超出电机范围，尤其绿球的腕关节；但紫哑铃在有越界动作的情况下也成功，因此越界是风险信号，不是单独成立的因果结论。

TCP 到**物体中心**的距离不是表面/可抓取区域距离，尤其不能据此断定长杆或哑铃的视觉定位精度。双指接触也可能是短暂的；应结合接触持续时间、抬升和录像判断。

## 复现

在服务器官方 OpenPI 环境下运行数据往返测试：

```bash
cd /root/autodl-tmp/workspace/openpi
.venv/bin/python /root/autodl-tmp/workspace/vla-manipulator/openpi_ext/check_so101_roundtrip.py \
  --dataset-root /root/autodl-tmp/datasets/lerobot_v1/HF2147552431/so101-mujoco-pick \
  --checkpoint /root/autodl-tmp/checkpoint_backups/pi05_so101_lora_so101_all_v1_step24999 \
  --index 41599
```

运行适配器测试：

```bash
cd /root/autodl-tmp/workspace/vla-manipulator/simulation
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m mj_env.scripts.check_adapter_roundtrip
```

后台运行闭环诊断时，为避免覆盖旧结果，应提供新的 `RUN_ID`：

```bash
cd /root/autodl-tmp/workspace/vla-manipulator
RUN_ID=so101_old25k_diagnosis_new screen -dmS so101_diag_new \
  bash scripts/diagnose_pi05_old25k_autodl.sh
```

本次两轮的服务器结果为 `/root/autodl-tmp/so101_old25k_diagnosis.jsonl` 与 `/root/autodl-tmp/so101_old25k_diagnosis_v2.jsonl`；两份原始记录、旧版 25k 及状态输入版 25k 的 max600 评估日志均已下载至本机桌面 `UVMS/SO101_25k_诊断记录` 文件夹，并逐文件核对 SHA-256。诊断记录只保存在本地，不随脚本推送 GitHub。数据和 checkpoint 未删除或上传；诊断结束后策略服务已停止。

# SO101 π0.5：状态 token 输入对照实验（已执行）

这份记录描述 2026-10-18 在 AutoDL **实际运行过**的实验，不是下一步计划。旧版 `pi05_so101_lora` 配置保持 `discrete_state_input=False`；新建独立配置 `pi05_so101_lora_state`，模型输入的核心差异是将该项改成 `True`。新配置还显式指向旧版归一化资产、将保存间隔改为 25,000 步，并使用独立实验名和 checkpoint 目录；其余沿用原来的 320 条 LeRobot 示教、LoRA 变体、batch size 8 和 action horizon 10。新实验从官方 `pi05_base` 开始训练，不是从旧版 25k checkpoint 续训。

服务器上曾运行 `scripts/run_state_25k_autodl.sh`，由 `scripts/train_pi05_so101_lora_state.sh` 调用 `openpi_ext/install_state_conditioned_config.py` 和 `openpi_ext/check_state_experiment.py`。状态文件报告 `TRAIN_25000_COMPLETE`；最终 checkpoint 位于 `/root/autodl-tmp/checkpoints/pi05_so101_lora_state/so101_state_true_25k_v1/24999`。训练日志最后一次每百步打印为 `Step 24900: loss=0.0014`，这只是训练 loss。

随后服务器运行了 `scripts/eval_state_25k_replan5_max600.sh`：`replan-steps=5`，8 类物体各 seed 1000–1019，共 160 条，每条最多 600 个 20 Hz 控制步（仿真时间 30 秒）。运行时用一个临时 `eval_no_ping.py` helper 关闭 WebSocket keepalive，以免 JAX 编译期间连接超时；仓库中保留了等价的 `simulation/mj_env/scripts/eval_no_ping.py`，使评估脚本不再依赖服务器根目录的临时文件。评估状态文件报告 `EVAL_COMPLETE`。

| 配置 | 闭环成功数 | 说明 |
| --- | ---: | --- |
| 旧版 25k，`discrete_state_input=False` | 88/160（55%） | 原 max600 评估日志 |
| 新版 25k，`discrete_state_input=True` | 64/160（40%） | 本次 max600 评估日志 |

这次对照中启用状态 token 的成绩更低；不能仅凭一次随机采样运行推断“状态 token 总会降低性能”。两次运行没有逐 episode 固定策略采样随机数，仍存在采样波动。训练 loss 也不能代替闭环成功率。原始评估日志已从服务器下载至本地 `SO101_25k_诊断记录` 文件夹；模型权重和正式数据没有随本次代码整理上传到 GitHub。

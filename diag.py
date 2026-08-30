"""诊断脚本:从已训练的 tb 日志提取关键 metrics 的时间演化,
用于定位训练崩塌的真实病因(UTD 不稳定 / ent_coef 异常 / 其他)。

不需要重跑,纯读取磁盘上现有 tb 日志,几十秒内输出诊断结论。
"""
import glob
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# 我们关心的 metrics 及其在 tb 日志里的 tag 名
TAGS = [
    ("rollout/ep_rew_mean",   "训练时 episode reward"),
    ("eval/mean_reward",      "评测 reward(5ep)"),
    ("train/ent_coef",        "熵系数 α"),
    ("train/ent_coef_loss",   "熵系数损失"),
    ("train/critic_loss",     "Critic loss"),
    ("train/actor_loss",      "Actor loss"),
]


def load_scalar(log_dir, tag):
    """读取 log_dir 下指定 tag 的全部 (step, value)。"""
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})  # 0 = 不限制
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def summarize(label, steps, values):
    if values is None or len(values) == 0:
        print(f"    {label:30s}: (无数据)")
        return
    nan_mask = np.isnan(values) | np.isinf(values)
    has_bad = nan_mask.any()
    finite = values[~nan_mask] if has_bad else values
    if len(finite) == 0:
        print(f"    {label:30s}: 全部 NaN/Inf!")
        return
    print(f"    {label:30s}: "
          f"初={finite[0]:.3g}  "
          f"峰={finite.max():.3g}@step{steps[~nan_mask][finite.argmax()]}  "
          f"谷={finite.min():.3g}@step{steps[~nan_mask][finite.argmin()]}  "
          f"末={finite[-1]:.3g}  "
          f"长度={len(finite)}"
          + ("  ★含NaN/Inf" if has_bad else ""))


def diagnose_one(log_dir):
    print(f"  日志: {log_dir}")
    # 1. 各指标基础统计
    for tag, label in TAGS:
        steps, values = load_scalar(log_dir, tag)
        summarize(label, steps, values)

    # 2. 评测 reward 的时间演化采样(10 个均匀采样点)
    steps, values = load_scalar(log_dir, "eval/mean_reward")
    if values is not None and len(values) >= 2:
        idx = np.linspace(0, len(values) - 1, min(10, len(values))).astype(int)
        print("    eval/mean_reward 演化采样:")
        for i in idx:
            print(f"        第{i+1:>2}次(step {int(steps[i]):>8}): reward={values[i]:>7.1f}")
        peak = values.argmax()
        print(f"    ★ 峰值在第 {peak+1}/{len(values)} 次(step {int(steps[peak])}); "
              f"之后 {len(values)-peak-1} 次评测全部低于此")

    # 3. ent_coef 关键时间点的值(诊断 ent_coef 异常)
    ent_steps, ent_vals = load_scalar(log_dir, "train/ent_coef")
    rew_steps, rew_vals = load_scalar(log_dir, "eval/mean_reward")
    if ent_vals is not None and rew_vals is not None and len(rew_vals) >= 2:
        peak_step = int(rew_steps[rew_vals.argmax()])
        # 找崩塌期(从峰之后到末尾)的 ent_coef 范围
        def nearest(target):
            i = int(np.argmin(np.abs(ent_steps - target)))
            return ent_vals[i]
        print(f"    ★ ent_coef 在关键时刻:")
        print(f"        训练开始       : {ent_vals[0]:.4f}")
        print(f"        reward 峰值时   : {nearest(peak_step):.4f}  (step {peak_step})")
        print(f"        训练结束时     : {ent_vals[-1]:.4f}  (step {int(ent_steps[-1])})")

    # 4. critic_loss 是否飞升(诊断 TQC 过估计崩塌)
    cs_steps, cs_vals = load_scalar(log_dir, "train/critic_loss")
    if cs_vals is not None and len(cs_vals) >= 4:
        early = np.median(cs_vals[: len(cs_vals) // 4])
        late = np.median(cs_vals[-len(cs_vals) // 4 :])
        ratio = late / early if early != 0 else float("inf")
        print(f"    ★ critic_loss 早期中位={early:.3g}, 末期中位={late:.3g}, "
              f"末期/早期={ratio:.2g}{'  ⚠ 显著飞升(疑似过估计崩塌)' if ratio > 5 else ''}")


def main():
    for seed in [42, 123]:
        print(f"\n========================== seed = {seed} ==========================")
        log_root = f"runs/tqc_utd1_seed{seed}/tb_logs"
        sub_dirs = sorted(glob.glob(os.path.join(log_root, "TQC_*")))
        if not sub_dirs:
            print(f"  无 tb 日志: {log_root}")
            continue
        for d in sub_dirs:
            diagnose_one(d)


if __name__ == "__main__":
    main()
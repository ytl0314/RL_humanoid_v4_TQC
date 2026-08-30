"""
    sb3-contrib 的 TQC 训练 Humanoid-v4  1分钟。

    执行  python train.py --seed 42 --output_dir runs/tqc_seed42

    每个 episode 4000 步 4000*0.015
"""
import argparse
import os
import re
import time
from pathlib import Path

from sb3_contrib import TQC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from utils.seeding import set_global_seeds
from utils.env_factory import (
    make_train_env, 
    make_eval_env_for_training,
    MAX_EPISODE_STEPS,
)
from utils.callbacks import SaveVecNormalizeCallback
from utils.progress_callback import RecordProgressCallback

GLOBAL_STEP_TARGET = 4_000_000 
DEFAULT_EVAL_FREQ = 200_000 #多少步评测一次


def parse_args():
    p = argparse.ArgumentParser(description="TQC 训练 Humanoid-v4 ")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_steps", type=int, default=GLOBAL_STEP_TARGET)
    p.add_argument("--output_dir", type=str, default="runs/tqc_default")
    p.add_argument("--eval_freq", type=int, default=DEFAULT_EVAL_FREQ)
    p.add_argument("--n_eval_episodes", type=int, default=5)
    p.add_argument("--checkpoint_freq", type=int, default=500_000) #保存checkpoints
    p.add_argument("--gradient_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--top_quantiles_to_drop_per_net", type=int, default=3)
    p.add_argument("--resume_from", type=str, default=None)
    return p.parse_args()

def _load_resume(model_zip: str, train_env):
    """
    CheckpointCallback 的命名约定:
        {prefix}_{steps}_steps.zip
        {prefix}_replay_buffer_{steps}_steps.pkl
        {prefix}_vecnormalize_{steps}_steps.pkl
    返回 (model, train_env)。
    """
    print(f"[Resume] 加载模型: {model_zip}")
    model = TQC.load(model_zip, env=train_env)

    m = re.match(r"(.*)_(\d+)_steps\.zip$", os.path.basename(model_zip))
    if not m:
        print("[Resume][warn] 无法解析 checkpoint 文件名, 跳过 replay buffer / vecnorm 加载")
        return model, train_env

    prefix, steps = m.group(1), m.group(2)
    base = os.path.dirname(model_zip)

    rb_path = os.path.join(base, f"{prefix}_replay_buffer_{steps}_steps.pkl")
    if os.path.exists(rb_path):
        model.load_replay_buffer(rb_path)
        print(f"[Resume] 已加载 replay buffer: {rb_path}")
    else:
        print(f"[Resume][warn] 未找到 replay buffer: {rb_path}(将以空 buffer 续训)")

    vn_path = os.path.join(base, f"{prefix}_vecnormalize_{steps}_steps.pkl")
    if os.path.exists(vn_path):
        from stable_baselines3.common.vec_env import VecNormalize
        restored = VecNormalize.load(vn_path, train_env.venv)
        model.set_env(restored)
        train_env = restored
        print(f"[Resume] 已加载 vecnormalize: {vn_path}")
    else:
        print(f"[Resume][warn] 未找到 vecnormalize: {vn_path}")

    return model, train_env


def main():
    args = parse_args()

    # 1  输出目录
    out = Path(args.output_dir)
    log_dir = out / "tb_logs"
    ckpt_dir = out / "checkpoints"
    best_dir = out / "best"
    mon_dir = out / "monitor"
    for d in (log_dir, ckpt_dir, best_dir, mon_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 2 固定种子
    set_global_seeds(args.seed)

    # 3 环境
    train_env = make_train_env(seed=args.seed, log_dir=str(mon_dir))
    eval_env = make_eval_env_for_training(seed=args.seed + 100_000)

    # 4.创建或加载模型
    if args.resume_from:
        model, train_env = _load_resume(args.resume_from, train_env)
        remaining = max(0, args.total_steps - model.num_timesteps)
        reset_flag = False
        total_to_run = remaining
        print(f"[Resume] 已训练 {model.num_timesteps} 步, 剩余 {remaining} 步")
    else:
        model = TQC(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.98,
            train_freq=1,
            gradient_steps=args.gradient_steps,                      # UTD, 默认 1
            top_quantiles_to_drop_per_net=args.top_quantiles_to_drop_per_net,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[512, 512]),                 # n_critics/n_quantiles 用默认
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=args.seed,
        )
        reset_flag = True
        total_to_run = args.total_steps

    # 5 回调  progress是训练时候看距离的 
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(ckpt_dir),
        name_prefix="tqc",
        save_replay_buffer=False,   # 关闭: 避免每次 checkpoint 存3GB buffer 把磁盘撑爆
        save_vecnormalize=True,     # 仅存归一化统计量
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(best_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
        # 仅在刷新 best 时保存一次对应的 vecnorm(避免每步存盘)
        callback_on_new_best=SaveVecNormalizeCallback(
            str(best_dir / "best_vecnormalize.pkl"), verbose=1),
    )
    
    progress_cb = RecordProgressCallback(record_freq=5000, verbose=1)
    callbacks = CallbackList([checkpoint_cb, eval_cb, progress_cb])
    

    # 6.  训练
    print("=" * 60)
    print(f"开始训练 TQC | seed={args.seed} | UTD={args.gradient_steps} "
          f"| net=[512,512] | run {total_to_run} steps")
    print(f"输出目录: {out}")
    print("=" * 60)

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_to_run,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=reset_flag,
        )
    except KeyboardInterrupt:
        print("\n[Interrupt] 训练被中断, 保存当前模型===========")

    print(f"\n训练完成, 耗时 {(time.time() - t0) / 3600:.2f} 小时")

    # 7.   保存最终模型
    final_model = out / "final_model.zip"
    final_vecnorm = out / "final_vecnormalize.pkl"
    model.save(str(final_model))
    train_env.save(str(final_vecnorm))

    print("\n已保存:")
    print(f"  final model   : {final_model}")
    print(f"  final vecnorm : {final_vecnorm}")
    print(f"  best  model   : {best_dir / 'best_model.zip'}")
    print(f"  best  vecnorm : {best_dir / 'best_vecnormalize.pkl'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
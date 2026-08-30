"""evaluate_and_record.py - 评测 best_model 并录制视频（最终修复版）。

用法:
    python evaluate_and_record.py [--n_episodes 3] [--seed 42]
"""
import argparse
import csv
import os
import time
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 环境配置
ENV_ID = "Humanoid-v4"
DT = 0.015
MAX_EPISODE_STEPS = 4001  #  4001，避免在 4000 时 reset
REPORT_STEP = 4000        # 在 4000 步时报告成绩

# 固定路径
MODEL_PATH = "runs/tqc_seed42/best/best_model.zip"
VECNORM_PATH = "runs/tqc_seed42/best/best_vecnormalize.pkl"
OUTPUT_DIR = "eval_results"


def parse_args():
    p = argparse.ArgumentParser(description="评测 TQC 模型并录制视频")
    p.add_argument("--n_episodes", type=int, default=3, help="评测多少个 episode")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--fps", type=int, default=67, help="视频帧率")
    return p.parse_args()


def main():
    args = parse_args()
    
    for pth in (MODEL_PATH, VECNORM_PATH):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"文件不存在: {pth}")
    
    out_dir = Path(OUTPUT_DIR)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("评测 TQC 模型 (1分钟最长距离)")
    print(f"  模型: {MODEL_PATH}")
    print(f"  Episode 数: {args.n_episodes}")
    print("=" * 60)
    
    def make_env():
        env = gym.make(
            ENV_ID,
            render_mode="rgb_array",
            terminate_when_unhealthy=False,
            max_episode_steps=MAX_EPISODE_STEPS,  # 4001
        )
        return env
    
    venv = DummyVecEnv([make_env])
    venv.seed(args.seed)
    venv = VecNormalize.load(VECNORM_PATH, venv)
    venv.training = False
    venv.norm_reward = False
    
    model = TQC.load(MODEL_PATH, env=venv)
    
    # 获取最底层 MuJoCo 环境
    mujoco_env = venv
    while hasattr(mujoco_env, 'venv'):
        mujoco_env = mujoco_env.venv
    while hasattr(mujoco_env, 'envs'):
        mujoco_env = mujoco_env.envs[0]
    while hasattr(mujoco_env, 'env') and mujoco_env.env is not None:
        mujoco_env = mujoco_env.env
    
    all_results = []
    
    for ep in range(args.n_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.n_episodes}")
        print(f"{'='*60}")
        
        obs = venv.reset()
        total_reward = 0
        steps = 0
        frames = []
        
        initial_x = float(mujoco_env.data.qpos[0])
        current_distance = 0.0
        final_distance = 0.0
        final_reward = 0.0
        final_height = 0.0
        
        print(f"初始位置: x = {initial_x:.2f} m")
        
        start_time = time.time()
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            
            steps += 1
            total_reward += float(reward[0])
            
            # 读取当前位置
            current_x = float(mujoco_env.data.qpos[0])
            current_distance = current_x - initial_x
            final_height = float(mujoco_env.data.qpos[2])
            
            # 录制视频帧
            try:
                frame = mujoco_env.render()
                if frame is not None:
                    frames.append(frame)
            except:
                pass
            
            # 打印进度
            if steps % 500 == 0:
                elapsed = steps * DT
                avg_speed = current_distance / elapsed if elapsed > 0 else 0
                print(f"  Step {steps:>4} | Time {elapsed:>5.1f}s | "
                      f"Distance {current_distance:>7.2f}m | "
                      f"AvgSpeed {avg_speed:>5.2f} m/s | "
                      f"Height {final_height:>4.2f}m")
            
            # ★ 在 4000 步时保存成绩
            if steps == REPORT_STEP:
                final_distance = current_distance
                final_reward = total_reward
                elapsed_60s = steps * DT
                avg_speed_60s = final_distance / elapsed_60s
                print(f"\n{'*'*60}")
                print(f"★ 60秒成绩 (Step {REPORT_STEP})")
                print(f"{'*'*60}")
                print(f"  前进距离: {final_distance:.2f} m")
                print(f"  平均速度: {avg_speed_60s:.3f} m/s")
                print(f"{'*'*60}\n")
            
            # 检查是否结束（4001 步）
            if bool(dones[0]):
                done = True
        
        # 使用 4000 步时保存的成绩
        avg_speed = final_distance / max(REPORT_STEP * DT, 0.001)
        
        print(f"\n  Episode {ep + 1} 完成!")
        print(f"  ★ 最终前进距离 (60秒): {final_distance:.2f} m")
        print(f"  ★ 平均速度: {avg_speed:.3f} m/s")
        
        # 保存视频
        if len(frames) > 0:
            video_path = video_dir / f"episode_{ep + 1}_distance_{final_distance:.2f}m.mp4"
            print(f"  正在保存视频: {video_path}")
            writer = imageio.get_writer(str(video_path), fps=args.fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"  视频已保存: {video_path}")
        
        all_results.append({
            'episode': ep + 1,
            'distance': final_distance,
            'avg_speed': avg_speed,
            'reward': final_reward,
            'steps': REPORT_STEP,
        })
    
    # 汇总
    print(f"\n{'='*60}")
    print("评测结果汇总 (60秒)")
    print(f"{'='*60}")
    
    distances = [r['distance'] for r in all_results]
    rewards = [r['reward'] for r in all_results]
    speeds = [r['avg_speed'] for r in all_results]
    
    print(f"  Episode 数: {len(all_results)}")
    print(f"  平均距离: {np.mean(distances):.2f} m")
    print(f"  标准差: {np.std(distances):.2f} m")
    print(f"  最大距离: {np.max(distances):.2f} m")
    print(f"  最小距离: {np.min(distances):.2f} m")
    print(f"  平均速度: {np.mean(speeds):.3f} m/s")
    print(f"  平均奖励: {np.mean(rewards):.1f}")
    print(f"{'='*60}")
    
    csv_path = out_dir / "eval_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'distance', 'avg_speed', 'reward', 'steps'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n结果已保存到: {csv_path}")
    
    venv.close()
    print("\n评测完成!")


if __name__ == "__main__":
    main()
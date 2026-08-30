# """
# 只看视频不保存
# 在 4000 步时输出 60 秒成绩。

#     c:\Users\Administrator\.conda\envs\human\python.exe d:/601/humanoid/humantqc_2/render_model.py --model_path runs/tqc_seed42/best/best_model.zip --vecnorm_path runs/tqc_seed42/best/best_vecnormalize.pkl --episodes 1
    
#     --model_path runs/tqc_seed42/checkpoints/tqc_4000000_steps.zip \
#                               --vecnorm_path runs/tqc_seed42/checkpoints/tqc_vecnormalize_4000000_steps.pkl \
#                               --episodes 1                      
# """
import argparse
import time
import os

import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 环境配置
ENV_ID = "Humanoid-v4"
DT = 0.015
MAX_EPISODE_STEPS = 4001       # 跑到 4500 步
REPORT_STEP = 4000             # 在 4000 步时报告成绩（60秒）


def parse_args():
    p = argparse.ArgumentParser(description="可视化观察 TQC 模型")
    p.add_argument("--fps", type=int, default=67, help="视频帧率")  # 改为 67
    p.add_argument("--model_path", required=True, help="模型 .zip 路径")
    p.add_argument("--vecnorm_path", required=True, help="VecNormalize .pkl 路径")
    p.add_argument("--episodes", type=int, default=1, help="渲染多少个 episode")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="使用确定性策略（默认）")
    p.add_argument("--slow_motion", type=float, default=1.0,
                   help="慢放倍数（1.0=正常速度，0.5=半速）")
    return p.parse_args()


def main():
    args = parse_args()
    
    for pth in (args.model_path, args.vecnorm_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"文件不存在: {pth}")
    
    print("=" * 60)
    print("模型可视化")
    print(f"  模型: {args.model_path}")
    print(f"  VecNormalize: {args.vecnorm_path}")
    print(f"  Episode 数: {args.episodes}")
    print("=" * 60)
    
    def make_env():
        env = gym.make(
            ENV_ID,
            render_mode="human",
            terminate_when_unhealthy=False,
            max_episode_steps=MAX_EPISODE_STEPS,  # 4500 步
        )
        return env
    
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(args.vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    
    model = TQC.load(args.model_path, env=venv)
    
    # 获取 MuJoCo 环境引用
    mujoco_env = venv.venv.envs[0]
    while hasattr(mujoco_env, 'env'):
        mujoco_env = mujoco_env.env
    
    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*60}")
        
        obs = venv.reset()
        done = False
        total_reward = 0
        steps = 0
        initial_x = None
        
        # 存储 4000 步时的数据
        report_distance = None
        report_reward = None
        report_avg_speed = None
        
        if hasattr(mujoco_env, 'data'):
            initial_x = float(mujoco_env.data.qpos[0])
            print(f"初始位置: x = {initial_x:.2f} m")
        
        start_time = time.time()
        
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = venv.step(action)
            
            # 先获取当前位置（在 reset 前）
            if hasattr(mujoco_env, 'data'):
                current_x = float(mujoco_env.data.qpos[0])
                current_vx = float(mujoco_env.data.qvel[0])
                distance = current_x - initial_x if initial_x is not None else 0
                height = float(mujoco_env.data.qpos[2])
            else:
                current_x = 0
                current_vx = 0
                distance = 0
                height = 0
            
            total_reward += float(reward[0])
            steps += 1
            done = bool(dones[0])
            
            # 实时打印（每 500 步）
            if steps % 500 == 0 and not done:
                elapsed = steps * DT
                avg_speed = distance / elapsed if elapsed > 0 else 0
                print(f"  Step {steps:>4} | Time {elapsed:>5.1f}s | "
                      f"Distance {distance:>7.2f}m | "
                      f"Speed {current_vx:>5.2f} m/s | "
                      f"AvgSpeed {avg_speed:>5.2f} m/s | "
                      f"Height {height:>4.2f}m | "
                      f"Reward {total_reward:>8.1f}")
            
            #  关键：在 4000 步时记录成绩
            if steps == REPORT_STEP:
                elapsed_60s = steps * DT
                avg_speed_60s = distance / elapsed_60s if elapsed_60s > 0 else 0
                report_distance = distance
                report_reward = total_reward
                report_avg_speed = avg_speed_60s
                
                print(f"\n{'*'*60}")
                print(f"★ 60秒成绩 (Step {REPORT_STEP})")
                print(f"{'*'*60}")
                print(f"  前进距离: {report_distance:.2f} m")
                print(f"  平均速度: {report_avg_speed:.3f} m/s")
                print(f"  总奖励: {report_reward:.1f}")
                print(f"  当前高度: {height:.2f} m")
                print(f"{'*'*60}\n")
            
            # 慢放控制
            if args.slow_motion != 1.0:
                time.sleep(DT * (1.0 / args.slow_motion - 1.0))
        
        # Episode 结束
        elapsed_real = time.time() - start_time
        
        print(f"\n  Episode 完成! (跑到 {steps} 步)")
        print(f"  {'='*50}")
        
        # 优先显示 4000 步的成绩
        if report_distance is not None:
            print(f"  【60秒官方成绩】")
            print(f"    前进距离: {report_distance:.2f} m")
            print(f"    平均速度: {report_avg_speed:.3f} m/s")
            print(f"    总奖励: {report_reward:.1f}")
            print(f"  {'='*50}")
        
        print(f"  渲染耗时: {elapsed_real:.1f} s")
    
    venv.close()
    print(f"\n{'='*60}")
    print("可视化结束")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
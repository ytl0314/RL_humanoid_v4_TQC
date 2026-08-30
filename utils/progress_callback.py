import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RecordProgressCallback(BaseCallback):
    """记录前进距离。
    
    1 分钟 = 4000 步, 
    - 每 100 步记录一次 x 坐标到 TensorBoard
    - 每 1000 步打印一次实时信息
    - 每 4000 步(1个episode)记录总距离
    """
    def __init__(self, record_freq=100, print_freq=1000, verbose=1):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.print_freq = print_freq
        self.episode_initial_x = None
        self.episode_count = 0
        
    def _on_training_start(self):
        self._init_mujoco_env()
        # 记录初始位置
        if hasattr(self.mujoco_env, 'data'):
            self.episode_initial_x = float(self.mujoco_env.data.qpos[0])
    
    def _on_step(self) -> bool:
        if not hasattr(self.mujoco_env, 'data'):
            return True
            
        current_x = float(self.mujoco_env.data.qpos[0])
        current_vx = float(self.mujoco_env.data.qvel[0])
        
        # 1. 高频记录到 TensorBoard
        if self.n_calls % self.record_freq == 0:
            self.logger.record('progress/x_position', current_x)
            self.logger.record('progress/x_velocity', current_vx)
        
        # 2. 打印实时信息
        if self.n_calls % self.print_freq == 0 and self.verbose > 0:
            print(f"\n[Step {self.num_timesteps:>8,}] "
                  f"x = {current_x:>8.2f} m, "
                  f"vx = {current_vx:>7.2f} m/s, "
                  f"elapsed = {(self.n_calls % 4000) * 0.015:.1f}s / 60s")
        
        # 3. Episode 结束 (每 4000 步)
        if self.n_calls % 4000 == 0 and self.episode_initial_x is not None:
            distance = current_x - self.episode_initial_x
            self.episode_count += 1
            
            # 记录到 TensorBoard
            self.logger.record('progress/episode_distance', distance)
            self.logger.record('progress/episode_count', self.episode_count)
            
            if self.verbose > 0:
                avg_speed = distance / 60.0  # 1分钟
                print(f"\n{'='*60}")
                print(f"[Episode {self.episode_count} 完成]")
                print(f"  前进距离: {distance:.2f} m")
                print(f"  平均速度: {avg_speed:.3f} m/s")
                print(f"  耗时: 60.0 s")
                print(f"{'='*60}")
            
            # 更新初始位置
            self.episode_initial_x = current_x
        
        return True
    
    def _init_mujoco_env(self):
       
        env = self.training_env
        while hasattr(env, 'venv'):
            env = env.venv
        if hasattr(env, 'envs'):
            self.mujoco_env = env.envs[0]
            if hasattr(self.mujoco_env, 'env'):
                self.mujoco_env = self.mujoco_env.env
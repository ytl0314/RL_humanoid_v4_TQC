import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ENV_ID = "Humanoid-v4"
EPISODE_DURATION = 60.0  # 秒
DT = 0.015               # Humanoid-v4 的物理时间步
MAX_EPISODE_STEPS = int(EPISODE_DURATION / DT)  # 4000 步


def make_train_env(seed: int, log_dir: str = None):
   
    def _init():
        env = gym.make(ENV_ID)
       
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = Monitor(env, log_dir)
        return env

    venv = DummyVecEnv([_init])
    venv.seed(seed)
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,   # 不归一化 reward
        clip_obs=10.0,
        gamma=0.99,
    )
    return venv


def make_eval_env_for_training(seed: int):
    def _init():
        env = gym.make(ENV_ID)
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([_init])
    venv.seed(seed)
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    venv.training = False
    return venv
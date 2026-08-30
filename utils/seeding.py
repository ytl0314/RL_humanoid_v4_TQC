"""全流程随机种子固定工具。

目标:覆盖环境初始化、模型初始化、数据采样等所有随机过程, 保证结果可复现。
本模块覆盖:Python random / NumPy / PyTorch(含 CUDA)/ Gym 环境的动作与观测空间采样。

注意:
- PyTorch 在 GPU 上的部分算子并非逐位确定, 因此"训练轨迹逐位复现"无法保证;
  但固定种子可保证统计意义上的可复现, 且"同机同版本重跑 evaluate.py"是可复现的。
"""
import os
import random

import numpy as np
from stable_baselines3.common.utils import set_random_seed


def set_global_seeds(seed: int) -> None:
    """设置 Python / NumPy / PyTorch / CUDA 层面的全局种子。

    set_random_seed(seed, using_cuda=True) 内部会调用
    torch.manual_seed / torch.cuda.manual_seed_all。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed, using_cuda=True)


def seed_env(env, seed: int) -> None:
    """种子化一个已创建环境(或 VecEnv)的动作/观测空间采样。

    env.reset(seed=...) 由调用方负责(SB3 VecEnv 通过 env.seed(seed) 设置,
    在下一次 reset 时生效)。
    """
    if hasattr(env, "action_space") and env.action_space is not None:
        env.action_space.seed(seed)
    if hasattr(env, "observation_space") and env.observation_space is not None:
        env.observation_space.seed(seed)
"""训练过程回调。

SaveVecNormalizeCallback:在 EvalCallback 刷新 best 模型时, 同步保存对应的
VecNormalize 统计量(均值/方差)。归一化统计量随模型一并保存,否则评测无效。

作为 EvalCallback(callback_on_new_best=...) 传入, 仅在出现新 best 时触发一次,
避免每步存盘拖慢训练。
"""
import os

from stable_baselines3.common.callbacks import BaseCallback


class SaveVecNormalizeCallback(BaseCallback):
    """在新 best 出现时, 保存当前训练环境的 VecNormalize 统计量到固定路径。"""

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(self.save_path)
            if self.verbose > 0:
                print(f"[SaveVecNormalize] new best -> {self.save_path}")
        return True
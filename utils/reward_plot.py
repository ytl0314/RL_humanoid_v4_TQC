import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# 路径
MONITOR_CSV = "runs/tqc_seed42/monitor/monitor.csv"
OUTPUT_DIR = "results"
MAX_STEPS = 4_000_000 


def load_monitor_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path, comment='#', names=['r', 'l', 't'])
    
    # 清洗数据
    df['r'] = pd.to_numeric(df['r'], errors='coerce')
    df['l'] = pd.to_numeric(df['l'], errors='coerce')
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    df = df.dropna(subset=['r', 'l', 't']).reset_index(drop=True)
    
    # 计算累计步数并过滤
    df['cumulative_steps'] = df['l'].cumsum()
    df = df[df['cumulative_steps'] <= MAX_STEPS].reset_index(drop=True)
    df['episode'] = range(1, len(df) + 1)
    
    return df


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"读取数据: {MONITOR_CSV}")
    df = load_monitor_data(MONITOR_CSV)
    
    if len(df) == 0:
        print("错误: 没有有效数据!")
        return
    
    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['r'], color='blue', linewidth=0.5)
    
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Reward Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "reward_curve.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 这个代码用于汇总分数

def parse_interval(interval_str):
    """解析区间字符串返回起始值用于排序"""
    match = re.findall(r"(\d+\.\d+)", interval_str)
    return float(match[0]) if match else 0.0

def plot_histogram(scores, title_str="", show_plot=True, save_path=None, counts_output_path=None):
    """
    绘制归一化复杂度得分的柱状图，并输出每个分数区间的数据个数
    """
    bin_interval = 0.05
    bins = np.arange(0, 1 + bin_interval, bin_interval)

    counts, bin_edges = np.histogram(scores, bins=bins)

    plt.figure(figsize=(12, 7))
    plt.hist(scores, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'ComplexityScore_norm Distribution\n{title_str}')
    plt.xlabel('ComplexityScore_norm')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(bins)
    plt.xlim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"直方图已保存到 {save_path}")

    if counts_output_path:
        with open(counts_output_path, 'w', encoding='utf-8') as f:
            f.write("分数区间及对应的数据个数：\n")
            for i in range(len(bins)-1):
                lower = bins[i]
                upper = bins[i+1]
                count = counts[i]
                interval = f"[{lower:.2f} - {upper:.2f}" + (")" if i != len(bins)-2 else "]")
                f.write(f"{interval}: {count}\n")

    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    input_folder = "./sig_score_0.05_test"
    output_dir = "./all_score"
    os.makedirs(output_dir, exist_ok=True)

    # 读取并汇总所有文件
    total_counts = defaultdict(int)
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("分数区间及对应的数据个数"):
                        continue
                    if ":" in line:
                        interval, count = line.split(":", 1)
                        total_counts[interval.strip()] += int(count.strip())

    # 生成scores数组
    scores = []
    for interval, count in total_counts.items():
        bounds = re.findall(r"\d+\.\d+", interval)
        if len(bounds) == 2:
            lower = float(bounds[0])
            upper = float(bounds[1])
            midpoint = (lower + upper) / 2
            scores.extend([midpoint] * count)

    # 保存汇总文件
    summary_path = os.path.join(output_dir, "all_scores_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("分数区间及对应的数据个数：\n")
        for interval in sorted(total_counts.keys(), key=parse_interval):
            f.write(f"{interval}: {total_counts[interval]}\n")

    # 绘制并保存直方图
    plot_histogram(
        scores=np.array(scores),
        title_str="Combined Results",
        show_plot=False,
        save_path=os.path.join(output_dir, "combined_histogram.png"),
        counts_output_path=os.path.join(output_dir, "histogram_counts.txt")
    )

if __name__ == "__main__":
    main()
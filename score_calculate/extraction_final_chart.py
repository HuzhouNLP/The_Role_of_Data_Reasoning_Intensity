import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 修改点：
# - 遍历文件夹下所有 .json 文件
# - 对每个文件分别读取并提取分数
# - 分别绘制柱状图
# - 分别保存区间统计结果到独立文件

folder_path = './sig_score_0.05_test'  # 指定包含 JSON 文件的文件夹

def read_json_file(file_path):
    """
    读取指定的 JSON 文件，并返回数据列表。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在。")
        return []
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的 JSON 文件。")
        return []

def extract_normalized_scores(data):
    """
    提取每个数据项的归一化复杂度得分，并确保分数在0到1之间。
    """
    scores = []
    for item in data:
        score = item.get('ComplexityScore_norm')
        if score is not None and 0 <= score <= 1:
            scores.append(score)
        else:
            # 若在实际场景中不想打印太多警告，可注释掉或做其他处理
            # print(f"警告：发现异常分数 {score}，已忽略。")
            pass
    return np.array(scores)

def save_counts_to_file(counts, bin_edges, output_path):
    """
    将分数区间及对应的数据个数保存到文本文件。
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("分数区间及对应的数据个数：\n")
            for i in range(len(counts)):
                if i < len(counts) - 1:
                    f.write(f"[{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}): {counts[i]}\n")
                else:
                    f.write(f"[{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}]: {counts[i]}\n")
        print(f"分数区间数据已保存到 {output_path}")
    except Exception as e:
        print(f"错误：无法保存分数区间数据到 {output_path}。原因：{e}")

def plot_histogram(scores, title_str="", show_plot=True, save_path=None, counts_output_path=None):
    """
    绘制归一化复杂度得分的柱状图，并输出每个分数区间的数据个数。

    参数：
    - scores: 提取到的分数（numpy array）
    - title_str: 图表标题（可附加文件名）
    - show_plot: 是否调用 plt.show() 显示图像
    - save_path: 如果提供，则把图保存到对应路径
    - counts_output_path: 如果提供，则把区间统计写到该文件
    """
    # 定义分数区间，间隔为0.05
    bin_interval = 0.05
    bins = np.arange(0, 1 + bin_interval, bin_interval)  # 0.00到1.00，共21个边界点

    # 计算每个区间的数量
    counts, bin_edges = np.histogram(scores, bins=bins)

    # 绘制柱状图
    plt.figure(figsize=(12, 7))
    plt.hist(scores, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'ComplexityScore_norm Distribution\n{title_str}')
    plt.xlabel('ComplexityScore_norm')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(bins)  # 设置x轴刻度为 0.00, 0.05, ..., 1.00
    plt.xlim(0, 1)    # 将横坐标限制在 0 到 1 之间
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"直方图已保存到 {save_path}")

    if show_plot:
        plt.show()
    else:
        # 若不显示，记得关闭图防止叠加
        plt.close()

    # 在控制台打印区间数量
    print("分数区间及对应的数据个数：")
    for i in range(len(counts)):
        # 左闭右开区间 except 最后一个区间是闭区间
        if i < len(counts) - 1:
            print(f"[{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}): {counts[i]}")
        else:
            print(f"[{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}]: {counts[i]}")

    # 保存分数区间数据到文件
    if counts_output_path:
        save_counts_to_file(counts, bin_edges, counts_output_path)

def main():
    # 获取文件夹中所有的 JSON 文件
    json_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
    if not json_files:
        print("指定文件夹中没有找到任何 JSON 文件。")
        return

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        data = read_json_file(file_path)
        if not data:
            print(f"文件 {json_file} 中没有读取到有效数据。")
            continue

        scores = extract_normalized_scores(data)
        if len(scores) == 0:
            print(f"文件 {json_file} 中没有有效的 ComplexityScore_norm 分数。")
            continue

        print(f"\n===== 文件: {json_file} =====")
        print(f"共读取到 {len(data)} 条记录, 其中 {len(scores)} 条包含有效的分数。")

        # 生成保存的文件名
        base_name = os.path.splitext(json_file)[0]  # 去掉后缀
        hist_img_name = f"{base_name}_hist.png"    # 直方图保存图像
        counts_txt_name = f"{base_name}_counts.txt" # 区间统计文本

        # 调用绘图并保存
        plot_histogram(
            scores,
            title_str=json_file,
            show_plot=False,  # 若不想弹窗显示图形，可设为False
            save_path=os.path.join(folder_path, hist_img_name),
            counts_output_path=os.path.join(folder_path, counts_txt_name)
        )

if __name__ == "__main__":
    main()

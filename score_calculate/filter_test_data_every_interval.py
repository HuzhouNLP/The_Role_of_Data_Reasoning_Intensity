import os
import json
import math

# 这个代码用于提取指定数量的数据，极大极小区间算作一个区间，中间区间按照步长计算
# 小于 0.2 区间定义为 极小区间，
# 大于 0.9 区间定义为 极大区间，

# ======================== 可修改参数 ========================
BIN_SIZE = 0.05           # 分数区间步长（中间区间）
MAX_PER_BIN = 80          # 每个区间最多提取多少条数据
INPUT_FOLDER = "./sig_score_test"         # 输入文件夹
OUTPUT_FOLDER = "./sig_score_0.05_test"   # 输出文件夹
SCORE_KEY = "ComplexityScore_norm"        # 分数对应的键名
# ==========================================================

def ensure_folder_exists(folder_path):
    """如果文件夹不存在，则创建它。"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def floor_score(score, decimal_places=2):
    """
    将分数截断到指定的小数位数，不进行四舍五入。
    例如，0.4178650589095069 -> 0.41
    """
    factor = 10 ** decimal_places
    return math.floor(score * factor) / factor

def get_bins(bin_size=0.05):
    """
    定义自定义分箱：
    - [0.0, 0.2)
    - [0.2, 0.25)
    - [0.25, 0.30)
    - ...
    - [0.85, 0.90)
    - [0.90, 1.00]

    返回一个列表，包含每个区间的 (lower, upper) 元组。
    """
    bins = []
    # 极小区间
    bins.append( (0.0, 0.2) )

    # 中间区间，每步0.05，从0.2到0.9
    current = 0.2
    while current < 0.9:
        next_edge = current + bin_size
        next_edge = round(next_edge, 2)  # 确保精度
        bins.append( (round(current, 2), next_edge) )
        current = next_edge

    # 极大区间
    bins.append( (0.9, 1.0) )

    return bins

def assign_bin(score, bins):
    """
    根据分数将数据分配到相应的区间。

    :param score: 分数值
    :param bins: 分箱列表
    :return: 区间索引，如果未找到返回None
    """
    for idx, (lower, upper) in enumerate(bins):
        if idx < len(bins) - 1:
            if lower <= score < upper:
                return idx
        else:
            if lower <= score <= upper:
                return idx
    return None

def process_file(input_path, output_path, bins, max_per_bin=20):
    """
    处理单个 JSON 文件：
    1. 读取数据
    2. 根据分数分配到不同区间
    3. 对每个区间提取最多 max_per_bin 条数据
    4. 将结果写入新的 JSON 文件

    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param bins: 分箱列表
    :param max_per_bin: 每个区间最多提取的数据条数
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"文件 {input_path} 读取失败：{e}")
        return

    if not isinstance(data, list):
        print(f"警告：文件 {input_path} 中顶层不是列表，将跳过处理。")
        return

    # 构建每个区间的数据列表
    bins_dict = {idx: [] for idx in range(len(bins))}
    total_valid_items = 0
    for item in data:
        score = item.get(SCORE_KEY)
        if isinstance(score, (int, float)) and 0 <= score <= 1:
            floored_score = floor_score(score, 2)
            bin_idx = assign_bin(floored_score, bins)
            if bin_idx is not None:
                bins_dict[bin_idx].append(item)
                total_valid_items += 1
        # 如果 score 无效或不在 [0,1]，则跳过

    # 打印每个区间的原始数据数量
    print(f"处理文件: {input_path}")
    print(f"总有效数据量: {total_valid_items}")
    print("每个区间的原始数据数量:")
    for idx, (lower, upper) in enumerate(bins):
        count = len(bins_dict.get(idx, []))
        print(f"  区间 [{lower:.2f}, {upper:.2f}{')' if idx < len(bins)-1 else ']'}: {count} 条")

    # 对每个区间提取最多 max_per_bin 条数据
    selected_bins = {}
    for idx in range(len(bins)):
        bin_data = bins_dict.get(idx, [])
        selected = bin_data[:max_per_bin]
        selected_bins[idx] = selected
        lower, upper = bins[idx]
        print(f"  提取区间 [{lower:.2f}, {upper:.2f}{')' if idx < len(bins)-1 else ']'}: 需要 {max_per_bin}, 实际取 {len(selected)}")

    # 合并所有选中的数据
    extracted_data = []
    for idx in range(len(bins)):
        extracted_data.extend(selected_bins.get(idx, []))

    # 打印最终每个区间的提取数量
    print("每个区间的最终提取数据数量:")
    for idx, (lower, upper) in enumerate(bins):
        count = len(selected_bins.get(idx, []))
        print(f"  区间 [{lower:.2f}, {upper:.2f}{')' if idx < len(bins)-1 else ']'}: {count} 条")

    # 检查是否达到预期总数
    expected_total = max_per_bin * len(bins)
    actual_total = len(extracted_data)
    print(f"预期提取总数: {expected_total}, 实际提取总数: {actual_total}")

    # 写入输出文件
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        print(f"已处理并输出: {output_path}, 共提取 {len(extracted_data)} 条数据.\n")
    except Exception as e:
        print(f"写入文件 {output_path} 时出错：{e}")

def main():
    ensure_folder_exists(OUTPUT_FOLDER)

    # 获取输入文件夹内所有 .json 文件
    try:
        all_files = os.listdir(INPUT_FOLDER)
    except FileNotFoundError:
        print(f"输入文件夹 {INPUT_FOLDER} 不存在。")
        return

    json_files = [f for f in all_files if f.lower().endswith(".json")]

    if not json_files:
        print(f"文件夹 {INPUT_FOLDER} 中未找到任何 JSON 文件。")
        return

    # 定义分箱
    bins = get_bins(bin_size=BIN_SIZE)

    for fname in json_files:
        input_path = os.path.join(INPUT_FOLDER, fname)
        output_path = os.path.join(OUTPUT_FOLDER, fname)
        process_file(
            input_path,
            output_path,
            bins=bins,
            max_per_bin=MAX_PER_BIN
        )

if __name__ == "__main__":
    main()

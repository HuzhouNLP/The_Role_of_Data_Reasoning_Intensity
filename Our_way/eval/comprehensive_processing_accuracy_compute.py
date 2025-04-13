import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path

# 定义多个数据集的输入目录
data_pairs = [
    {
        'input_directory': Path('./Reclor/')
    },
    {
        'input_directory': Path('./LogiQA2.0/')
    },
    {
        'input_directory': Path('./LogiQA/')
    },
    {
        'input_directory': Path('./LogicBench/')
    },
    {
        'input_directory': Path('./Multi_LogiEval/')
    },
]


def parse_label(label):
    """
    统一解析函数：
    - 若 label 是 int，直接返回 (MCQA)
    - 若 label 是字符串且可解析为 int，则转成 int (MCQA)
    - 若 label 是字符串且为 'yes'/'no'，则映射为 1/0 (BQA)
    - 否则返回 None
    """
    if label is None:
        return None

    # 如果已经是整数，直接返回
    if isinstance(label, int):
        return label

    # 如果是字符串，进行处理
    if isinstance(label, str):
        label_str = label.strip().lower()

        # 若能解析成整数
        if label_str.isdigit():
            return int(label_str)

        # 若是 yes/no 则映射为 1/0
        if label_str == 'yes':
            return 1
        elif label_str == 'no':
            return 0

    # 其他情况，统一返回 None
    return None


def compute_metrics(input_directory):
    """
    计算真实值和模型预测值的差距，计算准确率，召回率，F1分数等
    返回该输入目录下所有模型的评估结果列表
    """
    # 用于存储每个模型的评估结果
    metrics_results = []

    # 遍历根目录下的所有子文件夹
    for root in input_directory.rglob('generated_predictions.jsonl'):
        predictions_file = root
        root_dir = predictions_file.parent

        # 输出文件路径
        output_file = root_dir / 'predict_results.json'

        print(f"正在处理文件夹：{root_dir}")

        # 加载模型预测结果
        true_labels = []
        predicted_labels = []
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pred_data = json.loads(line)
                        true_label_str = pred_data.get('label')
                        predict_str = pred_data.get('predict')
                        if true_label_str is not None:
                            true_label_str = true_label_str.strip()
                        if predict_str is not None:
                            predict_str = predict_str.strip()
                        parsed_true_label = parse_label(true_label_str)
                        parsed_predict = parse_label(predict_str)
                        true_labels.append(parsed_true_label)
                        predicted_labels.append(parsed_predict)
                    except json.JSONDecodeError:
                        print(f"警告：在 {root_dir} 中发现无效的 JSON 行: {line}")
        except FileNotFoundError:
            print(f"警告：文件 {predictions_file} 未找到，跳过。")
            continue

        # 移除任何 None 值（以防解析错误）
        final_true_labels = []
        final_predicted_labels = []
        for t_label, p_label in zip(true_labels, predicted_labels):
            # 如果某一方是 None，表示无法解析为可用的数字/yes/no => 舍弃该条
            if t_label is not None and p_label is not None:
                final_true_labels.append(t_label)
                final_predicted_labels.append(p_label)

        # 检查是否有有效的标签
        if len(final_true_labels) == 0:
            print(f"在 {root_dir} 中未找到有效的标签，跳过。")
            continue

        # 计算评估指标
        accuracy = accuracy_score(final_true_labels, final_predicted_labels)

        precision, recall, f1, _ = precision_recall_fscore_support(
            final_true_labels, final_predicted_labels, average='macro', zero_division=0
        )

        # 准备结果字典
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # 将结果保存到 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"已在 {root_dir} 中计算指标并保存到 {output_file}")

        # 添加模型名称和结果到汇总列表
        model_name = root_dir.name
        results['model'] = model_name
        metrics_results.append(results)

    return metrics_results


def aggregate_results(metrics_list, aggregated_output_file):
    """
    将所有模型的评估结果汇总到一个 JSON 文件中
    """
    # 将所有结果写入到输出文件
    with open(aggregated_output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_list, f, ensure_ascii=False, indent=4)

    print(f"所有模型的结果已汇总并保存到 {aggregated_output_file}")


def main():
    for pair in data_pairs:
        input_directory = pair['input_directory']

        print(f"\n开始处理数据集：输入目录 = {input_directory}")

        # 步骤1：计算评估指标
        metrics = compute_metrics(input_directory)

        # 步骤2：汇总评估结果到每个输入目录下的 aggregated_results.json
        aggregated_output_file = input_directory / 'aggregated_results.json'
        aggregate_results(metrics, aggregated_output_file)

    print("\n所有数据集的处理已完成！")


if __name__ == "__main__":
    main()
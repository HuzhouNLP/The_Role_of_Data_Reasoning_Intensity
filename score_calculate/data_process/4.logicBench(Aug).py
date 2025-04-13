import os
import json

# 数据集名称
dataname = "LogicBench"
t_type = "BQA"  # 因为是二选一类型，设置为 "BQA"
r_type = ""  # 推理类型根据具体路径文件夹名判断

# 文件夹名称到推理类型的映射
reasoning_type_mapping = {
    "first_order_logic": "FOL",
    "nm_logic": "NM",
    "propositional_logic": "PL"
}


# 自动识别并提取内容的函数
def process_json_file(file_path, reasoning_type):
    processed_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 读取整个 JSON 文件

    # 遍历每个 data_sample
    for entry in data["data_samples"]:
        context = entry.get("context", "")
        qa_pairs = entry.get("qa_pairs", [])

        # 遍历 qa_pairs，把每个问题拆分成独立的条目
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")

            # 构建符合模板的字典
            data_item = {
                "id": None,  # 可以稍后根据需要填充
                "dataname": dataname,
                "task type": t_type,  # 任务类型是 BQA
                "reasoning type": reasoning_type,  # 推理类型来自文件夹映射
                "context": context,
                "question": question,
                "options": [],  # 因为是二选一问题，选项可以留空
                "answer": answer,  # yes 或 no
                "split": "train"  # 统一设置为 train
            }

            processed_items.append(data_item)

    return processed_items


# 根据文件路径自动判断推理类型
def determine_reasoning_type_from_path(path):
    for key in reasoning_type_mapping:
        if key in path:
            return reasoning_type_mapping[key]
    return ""  # 如果没有匹配，返回空字符串


# 遍历目录，找到所有json文件，并处理每个文件
def process_directory(directory_path):
    processed_data = []
    file_id = 1  # 用于为每个文件生成唯一的 id
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                # 从文件夹路径中判断推理类型
                reasoning_type = determine_reasoning_type_from_path(root)

                # 处理文件中的所有数据项
                items = process_json_file(file_path, reasoning_type)

                # 为每个数据项分配唯一的 id，并加入处理结果
                for item in items:
                    item['id'] = file_id
                    processed_data.append(item)
                    file_id += 1

    return processed_data


# 将处理后的数据保存为JSON文件
def save_to_json(data):
    # 获取当前代码所在的路径
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # 生成 output 文件夹路径
    output_directory = os.path.join(current_directory, "output")

    # 如果 output 文件夹不存在，创建它
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 保存文件到 output 文件夹中，文件名为 dataname.json
    output_file = os.path.join(output_directory, f"{dataname}.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"数据已保存到 {output_file}")


# 主函数
def main():
    # 输入目录
    directory_path = input("请输入要遍历的文件夹路径: ")

    # 处理文件夹中的数据
    processed_data = process_directory(directory_path)

    # 将数据保存为JSON文件
    save_to_json(processed_data)


if __name__ == "__main__":
    main()

import os
import json

# 数据集名称
dataname = "reclor"
t_type = "MCQA"  # 因为是多项选择题，设置为 "MCQA"
r_type = ""  # 推理类型可以根据具体需求填充
set_type = ""  # 如果无法确定手动填入 "train"、"dev" 或 "test"

# 自动识别并提取内容的函数
def process_json_file(file_path, split, file_id):
    processed_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 读取整个 JSON 文件

    for entry in data:
        context = entry.get("context", "")
        question = entry.get("question", "")
        options = entry.get("answers", [])
        label = entry.get("label", None)  # 测试集可能没有 label

        # 构建符合模板的字典
        data_item = {
            "id": file_id,  # 使用外部递增的 file_id
            "dataname": dataname,
            "task type": t_type,  # 任务类型是 MCQA
            "reasoning type": r_type,  # 推理类型可以手动设置
            "context": context,
            "question": question,
            "options": options[:4],  # 确保选项不超过 4 个
            "answer": label if label is not None else "",  # 如果有 label，则保留，否则填充 ""
            "split": split  # 自动从文件名判断 train, dev, test
        }

        processed_items.append(data_item)
        file_id += 1  # 递增 file_id

    return processed_items, file_id

# 根据文件名自动判断数据集拆分类型
def determine_split_from_filename(filename):
    if "train" in filename.lower():
        return "train"
    elif "val" in filename.lower():
        return "dev"
    elif "test" in filename.lower():
        return "test"
    else:
        return set_type

# 遍历目录，找到所有 json 文件，并处理每个文件
def process_directory(directory_path):
    processed_data = []
    file_id = 1  # 用于为每个文件生成唯一的 id
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                # 从文件名中判断 split
                split = determine_split_from_filename(file)

                # 处理文件中的所有数据项
                items, file_id = process_json_file(file_path, split, file_id)  # 传递递增的 file_id

                # 加入处理结果
                processed_data.extend(items)

    return processed_data

# 将处理后的数据保存为 JSON 文件
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

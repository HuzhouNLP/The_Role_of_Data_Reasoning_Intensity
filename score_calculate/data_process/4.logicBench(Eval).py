import os
import json

# 数据集名称
dataname = "FOLIO"
t_type = "BQA"  # 任务类型为 BQA
r_type = "FOL"  # 推理类型为 FOL
set_type = ""  # 如果无法确定手动填入 "train" 或 "test"

# 自动识别并提取内容的函数
def process_jsonl_file(file_path, split, start_id):
    processed_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            context = " ".join(entry.get("premises", []))
            question = entry.get("conclusion", "")
            label = entry.get("label", None)

            # 映射答案，True -> "yes", False -> "no", Uncertain -> "uncertain"
            answer_map = {"True": "yes", "False": "no", "Uncertain": "uncertain"}
            answer = answer_map.get(label, "")

            # 构建符合模板的字典
            data_item = {
                "id": start_id,  # 使用 start_id 作为唯一 ID
                "dataname": dataname,
                "task type": t_type,  # 任务类型是 BQA
                "reasoning type": r_type,  # 推理类型是 FOL
                "context": context,
                "question": question,
                "options": [],  # BQA 没有选项
                "answer": answer,  # 答案保留为文本形式：yes, no, uncertain
                "split": split  # 自动从文件名判断 train 或 test
            }

            processed_items.append(data_item)
            start_id += 1  # ID 自增

    return processed_items, start_id

# 根据文件名自动判断数据集拆分类型
def determine_split_from_filename(filename):
    if "train" in filename.lower():
        return "train"
    elif "validation" in filename.lower():
        return "test"
    else:
        return set_type  # 如果没有匹配，返回 set_type

# 遍历目录，找到所有 jsonl 文件，并处理每个文件
def process_directory(directory_path):
    processed_data = []
    file_id = 1  # 用于为每个文件生成唯一的 id，从1开始
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)

                # 从文件名中判断 split
                split = determine_split_from_filename(file)

                # 处理文件中的所有数据项，并更新 file_id
                items, file_id = process_jsonl_file(file_path, split, file_id)

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

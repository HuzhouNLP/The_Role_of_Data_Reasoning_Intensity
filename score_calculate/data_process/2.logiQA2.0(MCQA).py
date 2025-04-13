import os
import json

# 数据集名称
dataname = "LogiQA2.0"
r_type = ""  # 这里手动填写 "PL", "FOL", "NM", "NL"，"other"
t_type = "MCQA"  # 这里由文件中的type确定 "MCQA", "BQA", "NLI"
set_type = ""  # 如果无法确定手动填入"train"、"dev" 或 "test"


# 自动识别并提取内容的函数
def process_txt_file(file_path, split):
    processed_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # 读取所有行

    # 遍历每一行，将其视为一个独立的数据项
    for line in data:
        line = line.strip()
        if not line:
            continue

        try:
            # 解析整个行内容为 JSON 对象
            json_obj = json.loads(line)

            # 初始化字段
            question = ""
            context = ""
            options = []
            answer = ""

            # 提取背景信息（context）
            if "Context" in json_obj or "Text" in json_obj or "text" in json_obj or "context" in json_obj or "hypothesis" in json_obj:
                context = json_obj.get("text", "")

            # 提取问题
            if "Question" in json_obj or "premise" in json_obj or "question" in json_obj:
                question = json_obj.get("question", "")

            # 提取选项
            if "Options" in json_obj or "options" in json_obj:
                options = json_obj.get("options", [])

            # 提取答案（保持原来的格式，既可以是数字，也可以是其他类型）
            if "answer" in json_obj:
                answer = json_obj.get("answer", "")

            # 如果类型是BQA，options设为空
            if t_type == "BQA" or t_type == "NLI":
                options = []

            # 构建符合模板的字典
            data_item = {
                "id": None,  # 可以稍后根据需要填充
                "dataname": dataname,
                "task type": t_type,  # 这里由文件中的type确定 "MCQA", "BQA", "NLI"
                "reasoning type": r_type,  # 这里手动填写 "PL", "FOL", "NM", "other"
                "context": context,
                "question": question,
                "options": options[:4] if isinstance(options, list) else [],  # 确保选项不会超过4个
                "answer": answer,
                "split": split  # 自动从文件名判断 train, dev, test
            }

            # 将处理后的每个数据项加入列表
            processed_items.append(data_item)

        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")

    return processed_items  # 返回多个数据项


# 根据文件名自动判断数据集拆分类型
def determine_split_from_filename(filename):
    if "train" in filename.lower():
        return "train"
    elif "dev" in filename.lower():
        return "dev"
    elif "test" in filename.lower():
        return "test"
    else:
        return set_type  # 如果没有匹配，返回 set_type


# 遍历目录，找到所有txt文件，并处理每个文件
def process_directory(directory_path):
    processed_data = []
    file_id = 1  # 用于为每个文件生成唯一的 id
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                # 从文件名中判断 split
                split = determine_split_from_filename(file)

                # 处理文件中的所有数据项
                items = process_txt_file(file_path, split)

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

import os
import json

# 数据集名称
dataname = "LogiQA"
r_type = ""  # 这里手动填写 "PL", "FOL", "NM", "NL"，"other"
t_type = "MCQA"  # 这里由文件中的type确定 "MCQA", "BQA", "NLI"
set_type = ""  # 如果无法确定手动填入"train"、"dev" 或 "test"

# 映射答案字母到数字
answer_map = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3
}

# 自动识别并提取内容的函数
def process_txt_file(file_path, split):
    processed_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # 读取所有行

    i = 0
    while i < len(data):
        line = data[i].strip()

        if not line:
            i += 1
            continue

        # 初始化字段
        question = ""
        context = ""
        options = []
        answer = ""

        # 第1行是答案（小写字母）
        if len(line) == 1 and line.isalpha():
            answer = answer_map.get(line.lower(), "")
            i += 1
            line = data[i].strip()

        # 第2行是上下文
        context = line
        i += 1

        # 第3行是问题
        question = data[i].strip()
        i += 1

        # 第4-7行为选项
        options = []
        for _ in range(4):
            option = data[i].strip()
            # 去掉选项前面的 "A.", "B.", "C.", "D."
            if option.startswith(("A.", "B.", "C.", "D.")):
                option = option[2:].strip()
            options.append(option)
            i += 1

        # 构建符合模板的字典
        data_item = {
            "id": None,  # 可以稍后根据需要填充
            "dataname": dataname,
            "task type": t_type,  # 这里由文件中的type确定 "MCQA", "BQA", "NLI"
            "reasoning type": r_type,  # 这里手动填写 "PL", "FOL", "NM", "other"
            "context": context,
            "question": question,
            "options": options[:4] if isinstance(options, list) else [],  # 确保选项不会超过4个
            "answer": answer,  # 0,1,2,3形式的答案
            "split": split  # 自动从文件名判断 train, dev, test
        }

        # 将处理后的每个数据项加入列表
        processed_items.append(data_item)

        # 跳过空行
        if i < len(data) and not data[i].strip():
            i += 1

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

import os
import json

# 将制定文件夹下的所有jsonl文件转化为json文件

input_directory = './jsonl_error_correct'  # 输入 JSONL 文件的目录
output_directory = './json_error_correct'  # 输出 JSON 文件的目录

# 将目录下的所有 JSONL 文件转化为 JSON 文件
def convert_jsonl_directory_to_json(input_dir, output_dir):
    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录下的所有文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jsonl'):  # 只处理 .jsonl 文件
            jsonl_file = os.path.join(input_dir, file_name)
            json_file = os.path.join(output_dir, file_name.replace('.jsonl', '.json'))

            # 转换文件
            data = []
            with open(jsonl_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue  # 跳过无法解析的行

            with open(json_file, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, ensure_ascii=False, indent=4)

            print(f"Converted {jsonl_file} to {json_file}")

# 使用示例

convert_jsonl_directory_to_json(input_directory, output_directory)

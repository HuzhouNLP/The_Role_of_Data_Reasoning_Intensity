import os
import json

# 输入文件夹路径
input_folder = './all_test_data'
# 输出文件夹路径
output_folder = './all_score'
# 输出文件名
output_file_name = 'all_test_data.json'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 用于存储合并后的数据
merged_data = []

# 遍历输入文件夹中的所有文件
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 检查数据是否为列表
                    if isinstance(data, list):
                        merged_data.extend(data)
                    elif isinstance(data, dict):
                        merged_data.append(data)
                    else:
                        print(f"文件 {file_path} 中的数据格式不正确，既不是列表也不是字典。")
            except json.JSONDecodeError as e:
                print(f"读取文件 {file_path} 时出错: {e}")

# 输出文件路径
output_file_path = os.path.join(output_folder, output_file_name)

# 将合并后的数据写入输出文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"已成功将所有JSON文件合并并保存到 {output_file_path}")
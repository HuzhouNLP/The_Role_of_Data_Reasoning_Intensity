import os
import json

# =============== 可根据需要修改的路径 ===============
FOLDER_SRC = "./json_error_correct"  # 第一个文件夹（含 *_error.json）
FOLDER_DST = "./json"                # 第二个文件夹（要被覆盖/替换的文件）
# ==================================================

def ensure_folder_exists(folder_path):
    """若输出文件夹不存在，可根据需要创建，这里第二个文件夹一般是已存在的。"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_json_file(file_path):
    """读取 JSON 文件并返回其内容（可能是列表，也可能是单个 dict）。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"读取 {file_path} 出错: {e}")
        return None

def save_json_file(file_path, data):
    """将 data 写回 JSON 文件。"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"[保存] 成功写入 {file_path}")
    except Exception as e:
        print(f"[错误] 写入 {file_path} 失败: {e}")

def replace_item_in_target(target_data, new_item):
    """
    在 target_data (可能是列表或单个 dict) 中，根据 new_item['id'] 去匹配并替换。
    - 如果 target_data 是列表：寻找相同 id 的元素直接替换整个对象。
    - 如果 target_data 是单个 dict：若 id 相同，则直接覆盖。
    - 如果找不到相同 id，说明本身不存在，则可考虑是否要追加或忽略，这里默认忽略。
    """
    if not isinstance(new_item, dict):
        return target_data  # new_item 不符合预期，不做处理

    new_id = new_item.get("id")
    if new_id is None:
        return target_data  # new_item 没有 id，跳过

    if isinstance(target_data, dict):
        # 目标数据是单个对象 => 如果 id 相同就直接替换
        tgt_id = target_data.get("id")
        if tgt_id == new_id:
            return new_item  # 整个替换
        else:
            return target_data
    elif isinstance(target_data, list):
        # 目标数据是列表 => 查找同 id
        replaced = False
        for idx, obj in enumerate(target_data):
            if isinstance(obj, dict) and obj.get("id") == new_id:
                target_data[idx] = new_item  # 整个替换
                replaced = True
                break
        # 如果找不到对应 id，可以决定要不要 append，这里选择不做处理
        # if not replaced:
        #     target_data.append(new_item)
        return target_data
    else:
        # 如果既不是列表也不是 dict，就原样返回
        return target_data

def process_pair(src_file, dst_file):
    """
    对应文件做匹配并替换：
    1) 加载 src_file (xxx_error.json)
    2) 加载 dst_file (xxx.json)
    3) 对 src_file 中的每个条目(或单条)进行 id 匹配，替换到 dst_file 对应位置
    4) 将更新后的 dst_file 内容重新保存
    """
    src_data = load_json_file(src_file)
    dst_data = load_json_file(dst_file)
    if src_data is None or dst_data is None:
        print(f"[跳过] 因为无法读取: {src_file} 或 {dst_file}")
        return

    # 若 src_data 是列表，则逐条替换；若是单个对象，就只替换一次
    if isinstance(src_data, list):
        for item in src_data:
            dst_data = replace_item_in_target(dst_data, item)
    elif isinstance(src_data, dict):
        # 只有一个对象，直接替换
        dst_data = replace_item_in_target(dst_data, src_data)
    else:
        print(f"[警告] 源文件 {src_file} 顶层不是列表或字典，跳过。")
        return

    # 替换完以后，将 dst_data 写回原文件
    save_json_file(dst_file, dst_data)

def main():
    # 若目标文件夹不存在，可根据情况创建
    ensure_folder_exists(FOLDER_DST)

    # 获取源文件夹下所有文件，并过滤出 *_error.json
    all_src_files = os.listdir(FOLDER_SRC)
    error_json_files = [f for f in all_src_files if f.lower().endswith("_error.json")]

    if not error_json_files:
        print(f"文件夹 {FOLDER_SRC} 下未找到任何 *_error.json 文件。")
        return

    for error_fname in error_json_files:
        # 提取出不含 "_error" 的前半部分做匹配
        # 例如 "LogicBench_test_2020_error.json" => "LogicBench_test_2020"
        base_name = error_fname.rsplit("_error", 1)[0]  # 取最后一次出现的"_error"
        # 源文件完整路径
        src_file_path = os.path.join(FOLDER_SRC, error_fname)

        # 目标文件名（不含 "_error"），末尾加 .json
        # 如果原文件后缀不是 ".json" 而是别的，也要考虑 string replace
        # 这里默认 .json
        dst_fname = base_name + ".json"
        dst_file_path = os.path.join(FOLDER_DST, dst_fname)

        if not os.path.exists(dst_file_path):
            print(f"[警告] 匹配的目标文件 {dst_file_path} 不存在，跳过。")
            continue

        print(f"\n[处理] {src_file_path} => 替换到 => {dst_file_path}")
        process_pair(src_file_path, dst_file_path)

if __name__ == "__main__":
    main()

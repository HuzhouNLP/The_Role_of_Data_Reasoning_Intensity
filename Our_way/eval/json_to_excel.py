import os
import json
import pandas as pd

def json_to_excel(
    folder_list = ["Reclor", "LogiQA2.0", "LogiQA", "LogicBench","Multi_LogiEval"],
    output_excel = "results.xlsx"
):
    """
    读取给定文件夹列表中的 aggregated_results.json 文件，
    并将其写入同一个 Excel 文件的不同 Sheet 中。
    """

    # 创建一个 ExcelWriter 对象
    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        # 遍历每一个文件夹
        for folder in folder_list:
            json_path = os.path.join(folder, "aggregated_results.json")

            # 读取 JSON 文件
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # data 是一个包含多个 dict 的列表

            # 将 JSON 数据转换为 DataFrame
            df = pd.DataFrame(data)

            # 如果只需要固定列顺序，可以指定要输出的列
            # 注意：假设 JSON 中一定包含这几个字段
            df = df[["model", "accuracy", "precision", "recall", "f1_score"]]

            # 写入 Excel，sheet_name 就是对应数据集名称（文件夹名称）
            df.to_excel(writer, sheet_name=folder, index=False)

    print(f"Excel 文件已成功生成：{output_excel}")


if __name__ == "__main__":
    json_to_excel()

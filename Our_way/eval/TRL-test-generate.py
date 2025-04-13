import os
from jinja2 import Template

# 统一的 adapter 后缀
adapter_suffix = "filtered_no_repeat_output_36788_0.50_1.00_epoch16_no_eval_our_idea_weight"
# adapter 所在文件夹
adapter_dir = "./our_model_train_results/new_2.20_template_right/our_idea_weight"
# 保存的文件名字后缀（在adapter后面）# original_way   # our_idea_weight  # split_bucket
ways = ""
# 保存的文件名字前缀（一般只考虑给checkpoint做区分使用，平常为空即可，需要手动在后面加入 _）
check_points = ""
# 测试结果保存的根目录
output_dir = "./our_test_results/test_model_qujian"
test_data_dir = "./our_test_results"
# 定义数据集信息
datasets = [
    {
        "dataset_name": "LogicBench_test_2020.json",
        "output_prefix": "LogicBench",
    },
    {
        "dataset_name": "LogiQA_test_651.json",
        "output_prefix": "LogiQA",
    },
    {
        "dataset_name": "LogiQA2.0_test_1572.json",
        "output_prefix": "LogiQA2",
    },
    {
        "dataset_name": "Multi-LogiEval_1696.json",
        "output_prefix": "Multi_LogiEval",
    },
    {
        "dataset_name": "reclor_dev_500.json",
        "output_prefix": "Reclor",
    }
]

# 读取模板文件
with open("TRL-test-template.py", "r") as f:
    template_content = f.read()

template = Template(template_content)

# 生成每个数据集的配置文件
for ds in datasets:
    rendered = template.render(
        ADAPTER_SUFFIX=adapter_suffix,
        ADAPTER_DIR=adapter_dir,
        DATASET_NAME=ds["dataset_name"],
        OUTPUT_DIR=f"{output_dir}/{ds['output_prefix']}",
        TEST_DATA_DIR=test_data_dir,
        WAYS=ways,
        CHECK_POINTS=check_points
    )

    config_filename = f"config_{ds['output_prefix']}.py"
    with open(config_filename, "w") as f:
        f.write(rendered)
    print(f"已生成 {config_filename}")
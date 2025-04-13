import json
import transformers
import torch
from datasets import Dataset, Features, Value, Sequence
import os
# sageattention使用
# from sageattention import sageattn
# import torch.nn.functional as F
# F.scaled_dot_product_attention = sageattn

# 指定模型路径
model_id = "../llama3.1_70b"

# 创建 text-generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
#  device_map="auto"

# 设置 pad_token 和 pad_token_id
pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

# 读取 JSON 文件
input_file = 'data/reclor_3_id_change.json'  # 你的输入 JSON 文件路径
output_dir = 'data/reclor_data_back'          # 输出文件目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 指定特征
features = Features({
    'id': Value('int64'),              # 'id' 字段指定为整数类型
    'dataname': Value('string'),
    'task type': Value('string'),
    'reasoning type': Value('string'),
    'context': Value('string'),
    'question': Value('string'),
    'options': Sequence(Value('string')),
    'answer': Value('string'),
    'split': Value('string'),
    # 如果有其他字段，继续添加
})

# 创建 Dataset 对象
dataset = Dataset.from_list(data, features=features)

# 定义数据块的大小
chunk_size = 10000  # 每个数据块包含的数据量，可根据需要调整

# 计算总块数
total_chunks = (len(dataset) + chunk_size - 1) // chunk_size

# 检查已经处理的块，确定开始处理的块编号和起始 ID
processed_chunks = sorted([
    int(fname.split('_')[-1].split('.')[0])
    for fname in os.listdir(output_dir)
    if fname.startswith('output_part_') and fname.endswith('.jsonl')
])

if processed_chunks:
    last_chunk_idx = processed_chunks[-1]
    # 读取最后一个输出文件，获取已处理的最大 ID
    last_output_file = os.path.join(output_dir, f'output_part_{last_chunk_idx}.jsonl')
    max_processed_id = -1
    if os.path.exists(last_output_file):
        with open(last_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data_line = json.loads(line)
                    max_processed_id = max(max_processed_id, data_line['id'])
                except json.JSONDecodeError:
                    continue  # 跳过无法解析的行
    # 从下一个 ID 开始处理
    start_id = max_processed_id + 1
    start_chunk = last_chunk_idx
else:
    start_id = 0
    start_chunk = 0

print(f"Starting from chunk {start_chunk}, starting id {start_id}")

# 定义格式化消息的函数
def format_messages(example):
    context = example.get('context', '').strip()
    if not context:
        print(f"ID {example['id']} 的 context 提取失败或为空，请检查输入 JSON 文件。")
        return example  # 返回原始示例

    messages = [
        {
            "role": "system",
            "content": (
                "Instructions: Please extract the predicates and constants from the following context and create logical expressions that represent the relationships described. Format the output as a dictionary where each entry is a list of items. Follow these specific rules:\n"
                "1. **Extract predicates** as core action words or relationships defining connections between entities, and output them as a list under the 'Predicates' key.\n"
                "2. **Extract constants** as the specific entities or values mentioned in the context, and output them as a list under the 'Constants' key.\n"
                "3. **Create logical expressions** using the extracted predicates and constants. Each logical expression should be simple and based on a single predicate. Output them as a list under the 'Logical Expressions' key.\n"
                "4. Ensure that the output strictly follows the format provided in the example.\n"
                "5. Do not combine expressions using logical operators such as 'and,' 'or,' etc., unless the relationship is explicitly mentioned in the context.\n"
            )
        },
        {
            "role": "user",
            "content": (
                "### Example:\n\n"
                "{Context: \n"
                "\"If an individual consumes a significant amount of water, they will experience a state of hydration. "
                "Conversely, if excessive amounts of sugar are ingested, a sugar crash will ensue. It is known that at "
                "least one of the following statements is true: either Jane consumes ample water or she will not "
                "experience a sugar crash. However, the actual veracity of either statement remains ambiguous, as it "
                "could be the case that only the first statement is true, only the second statement is true, or both "
                "statements are true.\"\n}\n\n"
                "{Output:\n"
                "{\n"
                "    \"Predicates\": [\n"
                "        \"`Consumes(x, y)`: Represents the act of 'x' consuming 'y' (e.g., an individual consuming water or sugar).\",\n"
                "        \"`ExperienceState(x, y)`: Represents 'x' experiencing a state 'y' (e.g., hydration).\",\n"
                "        \"`Ingested(x, y)`: Represents 'x' ingesting 'y' (e.g., excessive sugar).\",\n"
                "        \"`Ensue(x)`: Represents that a condition 'x' follows or results (e.g., a sugar crash).\",\n"
                "        \"`TrueStatement(x)`: Indicates that 'x' is known to be true.\",\n"
                "        \"`NotExperience(x, y)`: Represents 'x' not experiencing a condition 'y' (e.g., not experiencing a sugar crash).\"\n"
                "    ],\n"
                "    \"Constants\": [\n"
                "        \"`Individual`: Represents a generic person in the context.\",\n"
                "        \"`Water`: The substance being consumed by an individual.\",\n"
                "        \"`Hydration`: The state that results from sufficient water consumption.\",\n"
                "        \"`Sugar`: A substance that can be ingested.\",\n"
                "        \"`SugarCrash`: The state that follows excessive sugar intake.\",\n"
                "        \"`Jane`: A specific person mentioned in the context.\"\n"
                "    ],\n"
                "    \"Logical Expressions\": [\n"
                "         \"`Consumes(Individual, Water)`\",\n"
                "         \"`ExperienceState(Individual, Hydration)`\",\n"
                "         \"`Ingested(Individual, Sugar)`\",\n"
                "         \"`Ensue(SugarCrash)`\",\n"
                "         \"`Consumes(Jane, Water)`\",\n"
                "         \"`NotExperience(Jane, SugarCrash)`\",\n"
                "         \"`TrueStatement(Consumes(Jane, Water) ∨ NotExperience(Jane, SugarCrash))`\"\n"
                "    ]\n"
                "}\n}\n\n"
                "---\n\n"
                "### Tips for Extracting Predicates, Constants, and Logical Expressions:\n"
                "- Focus on identifying core action or relational words for predicates.\n"
                "- Extract constants as the specific entities mentioned.\n"
                "- Use variables (`x`, `y`, etc.) to generalize when needed.\n"
                "- Ensure logical expressions are complete and accurately reflect relationships.\n\n"
                "### Your Task:\n"
                f"{{Context: \"{context}\"}}"
            )
        }
    ]

    # 将消息格式化为字符串
    formatted = ""
    for message in messages:
        role = message['role']
        content = message['content']
        if role == 'system':
            formatted += f"System: {content}\n\n"
        elif role == 'user':
            formatted += f"User: {content}\n\n"
    formatted += "Assistant:"

    # 将 prompt 添加到示例中，而不是替换示例
    example['prompt'] = formatted
    return example

# 定义解析输出的函数
def parse_output(example, output):
    # 检查 output 的类型
    if isinstance(output, list):
        if len(output) > 0 and isinstance(output[0], dict):
            assistant_content = output[0].get('generated_text', '').strip()
        else:
            print(f"Example ID {example['id']}: output list is empty or contains non-dict elements.")
            assistant_content = ''
    elif isinstance(output, dict):
        assistant_content = output.get('generated_text', '').strip()
    else:
        print(f"Example ID {example['id']}: output is neither list nor dict.")
        assistant_content = ''

    if not assistant_content:
        print(f"Example ID {example['id']}: assistant content is empty.")
        return example  # 或者进行其他处理，例如跳过该示例

    # 继续解析 assistant_content，提取 Predicates、Constants、Logical Expressions
    import re

    # 使用正则表达式匹配各部分
    predicates_match = re.search(r'"?Predicates"?\s*:\s*(\[[^\]]*\])', assistant_content, re.IGNORECASE)
    constants_match = re.search(r'"?Constants"?\s*:\s*(\[[^\]]*\])', assistant_content, re.IGNORECASE)
    logical_expressions_match = re.search(r'"?Logical Expressions"?\s*:\s*(\[[^\]]*\])', assistant_content, re.IGNORECASE)

    def parse_list_from_string(list_str):
        # 去掉首尾的方括号和空格
        list_str = list_str.strip()[1:-1].strip()
        # 分割列表项
        items = re.findall(r'"([^"]+)"', list_str)
        return items

    if predicates_match and constants_match and logical_expressions_match:
        predicates_list = parse_list_from_string(predicates_match.group(1))
        constants_list = parse_list_from_string(constants_match.group(1))
        logical_expressions_list = parse_list_from_string(logical_expressions_match.group(1))
    else:
        print(f"Example ID {example['id']}: Failed to parse assistant content.")
        predicates_list = []
        constants_list = []
        logical_expressions_list = []
        # 如果需要，可以将 assistant_content 保存以供调试
        example['assistant_content'] = assistant_content

    # 将解析结果添加到示例中
    example['Predicates'] = predicates_list
    example['Constants'] = constants_list
    example['Logical Expressions'] = logical_expressions_list

    # 移除 prompt 字段
    if 'prompt' in example:
        del example['prompt']

    return example

# 遍历数据块
for chunk_idx in range(start_chunk, total_chunks):
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, len(dataset))
    print(f"Processing chunk {chunk_idx}, data index from {start_idx} to {end_idx}")

    # 提取当前数据块
    chunk_dataset = dataset.select(range(start_idx, end_idx))

#    # 过滤掉 'split' 不为 'train' 的示例
#    chunk_dataset = chunk_dataset.filter(lambda x: x['split'] == 'train')

    # 过滤掉 ID 小于 start_id 的示例
    chunk_dataset = chunk_dataset.filter(lambda x: x['id'] >= start_id)

    # 对数据块进行预处理
    chunk_dataset = chunk_dataset.map(format_messages)

    # 过滤掉 prompt 为空的示例
    chunk_dataset = chunk_dataset.filter(lambda x: x['prompt'] != '')

    # 准备输出文件路径
    output_file = os.path.join(output_dir, f'output_part_{chunk_idx}.jsonl')

    # 获取已处理的 ID 集合
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data_line = json.loads(line)
                    processed_ids.add(data_line['id'])
                except json.JSONDecodeError:
                    continue  # 跳过无法解析的行
        print(f"Found {len(processed_ids)} existing examples in {output_file}")

    # 获取待处理的示例
    examples_to_process = [ex for ex in chunk_dataset if ex['id'] not in processed_ids]

    if not examples_to_process:
        print(f"No new examples to process in chunk {chunk_idx}.")
        start_id = 0  # 重置 start_id
        continue  # 跳过当前数据块

    # 定义批处理大小
    batch_size = 128  # 每处理 256 条数据保存一次结果

    # 将待处理的示例分成多个批次
    total_examples = len(examples_to_process)
    num_batches = (total_examples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_examples)
        batch_examples = examples_to_process[batch_start:batch_end]
        batch_prompts = [ex['prompt'] for ex in batch_examples]

        # 调用 pipeline 进行生成
        try:
            outputs = pipeline(
                batch_prompts,
                max_new_tokens=2048,  # 根据需要调整
                return_full_text=False,
                batch_size=16  # 这里的 batch_size 可以根据显存调整
            )
        except Exception as e:
            print(f"Error during generation in chunk {chunk_idx}, batch {batch_idx}: {e}")
            continue  # 跳过当前批次，继续下一个

        # 处理输出结果并逐条保存
        with open(output_file, 'a', encoding='utf-8') as f:
            for ex, output in zip(batch_examples, outputs):
                processed_example = parse_output(ex, output)
                json_line = json.dumps(processed_example, ensure_ascii=False)
                f.write(json_line + '\n')

        print(f"Saved batch {batch_idx + 1}/{num_batches} of chunk {chunk_idx} to {output_file}")

    print(f"Chunk {chunk_idx} processed and saved to {output_file}")

    # 重置 start_id，为下一个数据块准备
    start_id = 0  # 下一个数据块从头开始

print("All data processing completed.")
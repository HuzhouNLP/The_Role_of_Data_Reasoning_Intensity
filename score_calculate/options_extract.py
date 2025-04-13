import json
import transformers
import torch
from datasets import Dataset, Features, Value, Sequence
import os
import re
import copy

# 指定模型路径
model_id = "../llama3.1_70b"

# 创建 text-generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# 设置 pad_token 和 pad_token_id
pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

# 读取 JSON 文件
input_file = 'data/option_data_test.json'  # 你的输入 JSON 文件路径
output_dir = 'data/output_test'           # 输出文件目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 指定特征，添加所有必要字段
features = Features({
    'id': Value('int64'),              # 'id' 字段指定为整数类型
    'dataname': Value('string'),
    'task type': Value('string'),      # "MCQA" 或 "BQA"
    'reasoning type': Value('string'),
    'context': Value('string'),
    'question': Value('string'),
    'options': Sequence(Value('string')),  # 对于 BQA，可能为空
    'answer': Value('string'),
    'split': Value('string'),
    'Predicates': Sequence(Value('string')),           # 新增字段
    'Constants': Sequence(Value('string')),            # 新增字段
    'Logical Expressions': Sequence(Value('string')),  # 新增字段
    'Normalized Complexity Score': Value('float'),     # 新增字段
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

# =========================
# 修正后的模板定义
# =========================
messages_MCQA = [
    {
        "role": "system",
        "content": (
            "Instructions: Please analyze the following multiple-choice question data. For each option, extract the relevant preconditions, define the deduction target, outline the deduction steps based on the provided Predicates, Constants, and Logical Expressions, and determine whether the option is correct based on the given answer index. Follow these specific rules:\n"
            "1. **Output** should be a JSON object containing only the `option_analysis` field, which is a list of analyses for each option.\n"
            "2. For each option in option_analysis, include the following fields:\n"
            "   - option_index: The index of the option (0-based).\n"
            "   - option_text: The text of the option.\n"
            "   - preconditions: A list of relevant preconditions from the Logical Expressions that pertain to the option.\n"
            "   - deduction_target: The abstracted logical conclusion that the option is attempting to establish.\n"
            "   - deduction_steps: A step-by-step logical deduction process from the preconditions to the deduction target. Each step should include:\n"
            "     - step: The step number.\n"
            "     - task: A description of what is being checked or inferred in this step.\n"
            "     - expression: The logical expression used in this step, enclosed in backticks.\n"
            "     - result: The outcome of this step, enclosed in backticks.\n"
            "     - If a deduction step cannot proceed due to unsupported premises, indicate the failure and terminate further steps for that option.\n"
            "   - is_correct: A boolean indicating whether the option is correct (true) or not (false). The values of 'Answer' are '0', '1', '2', '3', where '0' represents the first option, '1' represents the second option, and so on. If the value of 'option_index' is the same as the value of 'Answer', then the value of 'is_correct' is 'true'\n"
            "3. **Format Requirements**:\n"
            "   - The output must strictly follow the JSON structure as shown in the example.\n"
            "   - Ensure consistency in field naming and hierarchy.\n"
            "   - Do not include any additional fields or information not specified in the example.\n"
            "4. **Important Considerations**:\n"
            "   - Only include preconditions that are directly relevant to the option being analyzed.\n"
            "   - Maintain logical rigor in deduction steps, ensuring each step follows from the previous ones based on the preconditions.\n"
            "   - Avoid including unrelated preconditions to minimize complexity and enhance clarity.\n"
        )
    },
    {
        "role": "user",
        "content": (
            "{\n"
            "    \"Example\": {\n"
            "        \"context\": \"In rheumatoid arthritis, the body's immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.\",\n"
            "        \"question\": \"The statements above, if true, most strongly support which one of the following conclusions?\",\n"
            "        \"options\": [\n"
            "            \"Unlike aspirin and other medications that reduce pain and swelling and that are currently available, the new medication would repair existing cell damage that had been caused by rheumatoid arthritis.\",\n"
            "            \"A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it.\",\n"
            "            \"Joint diseases other than rheumatoid arthritis would not be affected by the new medication.\",\n"
            "            \"The benefits to rheumatoid arthritis sufferers of the new medication would outweigh the medication's possible harmful side effects.\"\n"
            "        ],\n"
            "        \"answer\": \"1\",\n"
            "        \"Predicates\": [\n"
            "            \"Attack(x, y): Represents 'x' attacking 'y' (e.g., the immune system attacking healthy cells).\",\n"
            "            \"Release(x, y): Represents 'x' releasing 'y' (e.g., the release of a hormone).\",\n"
            "            \"Cause(x, y): Represents 'x' causing 'y' (e.g., the hormone causing pain and swelling).\",\n"
            "            \"Activate(x, y): Represents 'x' activating 'y' (e.g., the hormone being activated by injury or infection).\",\n"
            "            \"Inhibit(x, y): Represents 'x' inhibiting 'y' (e.g., the protein inhibiting the hormone).\",\n"
            "            \"Contain(x, y): Represents 'x' containing 'y' (e.g., the medication containing a protein).\"\n"
            "        ],\n"
            "        \"Constants\": [\n"
            "            \"ImmuneSystem: The body's defense mechanism.\",\n"
            "            \"HealthyCells: Cells in the joints that are not diseased.\",\n"
            "            \"Hormone: A chemical messenger involved in causing pain and swelling.\",\n"
            "            \"Pain: A sensation caused by the hormone.\",\n"
            "            \"Swelling: A condition caused by the hormone.\",\n"
            "            \"Injury: A condition that normally activates the hormone.\",\n"
            "            \"Infection: A condition that normally activates the hormone.\",\n"
            "            \"ArthritisMedication: A new medication for treating arthritis.\",\n"
            "            \"Protein: A component of the medication that inhibits the hormone.\"\n"
            "        ],\n"
            "        \"Logical Expressions\": [\n"
            "            \"Attack(ImmuneSystem, HealthyCells)\",\n"
            "            \"Release(ImmuneSystem, Hormone)\",\n"
            "            \"Cause(Hormone, Pain)\",\n"
            "            \"Cause(Hormone, Swelling)\",\n"
            "            \"Activate(Injury, Hormone)\",\n"
            "            \"Activate(Infection, Hormone)\",\n"
            "            \"Inhibit(Protein, Hormone)\",\n"
            "            \"Contain(ArthritisMedication, Protein)\"\n"
            "        ],\n"
            "    },\n\n"
            "    \"Output\": {\n"
            "        \"option_analysis\": [\n"
            "            {\n"
            "                \"option_index\": 0,\n"
            "                \"option_text\": \"Unlike aspirin and other medications that reduce pain and swelling and that are currently available, the new medication would repair existing cell damage that had been caused by rheumatoid arthritis.\",\n"
            "                \"preconditions\": [\n"
            "                    \"(1) Attack(ImmuneSystem, HealthyCells)\",\n"
            "                    \"(2) Release(ImmuneSystem, Hormone)\",\n"
            "                    \"(3) Cause(Hormone, Pain)\",\n"
            "                    \"(4) Cause(Hormone, Swelling)\"\n"
            "                ],\n"
            "                \"deduction_target\": \"Repair(ArthritisMedication, HealthyCellsDamage)\",\n"
            "                \"deduction_steps\": [\n"
            "                    {\n"
            "                        \"step\": \"1\",\n"
            "                        \"task\": \"Check if Attack(ImmuneSystem, HealthyCells) implies Damage(HealthyCells).\",\n"
            "                        \"expression\": \"Attack(ImmuneSystem, HealthyCells) → Damage(HealthyCells)\",\n"
            "                        \"result\": \"Supported by context (immune system attacking healthy cells causes damage).\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"2\",\n"
            "                        \"task\": \"Check if Contain(ArthritisMedication, Protein) and Inhibit(Protein, Hormone) imply Repair(ArthritisMedication, HealthyCellsDamage).\",\n"
            "                        \"expression\": \"Contain(ArthritisMedication, Protein) ∧ Inhibit(Protein, Hormone) → Repair(ArthritisMedication, HealthyCellsDamage)\",\n"
            "                        \"result\": \"Not supported. The context only states that the protein inhibits the hormone, not that it repairs damage.\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"3\",\n"
            "                        \"task\": \"Derivation fails.\",\n"
            "                        \"expression\": \"Derivation cannot proceed.\",\n"
            "                        \"result\": \"Repair(ArthritisMedication, HealthyCellsDamage) cannot be derived from the given preconditions.\"\n"
            "                    }\n"
            "                ],\n"
            "                \"is_correct\": false\n"
            "            },\n"
            "            {\n"
            "                \"option_index\": 1,\n"
            "                \"option_text\": \"A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it.\",\n"
            "                \"preconditions\": [\n"
            "                    \"(3) Cause(Hormone, Pain)\",\n"
            "                    \"(5) Activate(Injury, Hormone)\",\n"
            "                    \"(7) Inhibit(Protein, Hormone)\",\n"
            "                    \"(8) Contain(ArthritisMedication, Protein)\"\n"
            "                ],\n"
            "                \"deduction_target\": \"∃Patient, Injury : [Sustain(Patient, Injury) ∧ Unaware(Patient, Injury)]\",\n"
            "                \"deduction_steps\": [\n"
            "                    {\n"
            "                        \"step\": \"1\",\n"
            "                        \"task\": \"Determine the effect of Inhibit(Protein, Hormone) from the medication.\",\n"
            "                        \"expression\": \"Inhibit(Protein, Hormone)\",\n"
            "                        \"result\": \"Supported by context: The protein inhibits the hormone that causes pain and swelling.\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"2\",\n"
            "                        \"task\": \"Analyze the implication of inhibiting the hormone on pain and swelling.\",\n"
            "                        \"expression\": \"Inhibit(Protein, Hormone) → ¬Cause(Hormone, Pain) ∧ ¬Cause(Hormone, Swelling)\",\n"
            "                        \"result\": \"Supported by context: If the hormone is inhibited, it cannot cause pain and swelling.\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"3\",\n"
            "                        \"task\": \"Infer the patient's awareness of injury when pain and swelling are absent.\",\n"
            "                        \"expression\": \"¬Cause(Hormone, Pain) ∧ ¬Cause(Hormone, Swelling) → Unaware(Patient, Injury)\",\n"
            "                        \"result\": \"Supported by context: Without pain and swelling, the patient may not be aware of sustaining an injury.\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"4\",\n"
            "                        \"task\": \"Combine the above implications to conclude the deduction target.\",\n"
            "                        \"expression\": \"∃Patient, Injury : [Sustain(Patient, Injury) ∧ Unaware(Patient, Injury)]\",\n"
            "                        \"result\": \"Deduction is valid based on the inhibited hormone preventing awareness of injury.\"\n"
            "                    }\n"
            "                ],\n"
            "                \"is_correct\": true\n"
            "            },\n"
            "            {\n"
            "                \"option_index\": 2,\n"
            "                \"option_text\": \"Joint diseases other than rheumatoid arthritis would not be affected by the new medication.\",\n"
            "                \"preconditions\": [\n"
            "                    \"(7) Inhibit(Protein, Hormone)\",\n"
            "                    \"(8) Contain(ArthritisMedication, Protein)\"\n"
            "                ],\n"
            "                \"deduction_target\": \"∀x [JointDisease(x) ∧ x ≠ RheumatoidArthritis → ¬Affect(ArthritisMedication, x)]\",\n"
            "                \"deduction_steps\": [\n"
            "                    {\n"
            "                        \"step\": \"1\",\n"
            "                        \"task\": \"Check if the context provides information about other joint diseases.\",\n"
            "                        \"expression\": \"JointDisease(x) ∧ x ≠ RheumatoidArthritis\",\n"
            "                        \"result\": \"Not supported. The context only discusses rheumatoid arthritis.\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"2\",\n"
            "                        \"task\": \"Determine if there is any implication that the medication specifically targets rheumatoid arthritis.\",\n"
            "                        \"expression\": \"Contain(ArthritisMedication, Protein) → SpecificEffect(RheumatoidArthritis)\",\n"
            "                        \"result\": \"Not supported. The context does not specify that the protein exclusively affects rheumatoid arthritis.\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"3\",\n"
            "                        \"task\": \"Derivation fails.\",\n"
            "                        \"expression\": \"Derivation cannot proceed.\",\n"
            "                        \"result\": \"Cannot conclude that the medication does not affect other joint diseases.\"\n"
            "                    }\n"
            "                ],\n"
            "                \"is_correct\": false\n"
            "            },\n"
            "            {\n"
            "                \"option_index\": 3,\n"
            "                \"option_text\": \"The benefits to rheumatoid arthritis sufferers of the new medication would outweigh the medication's possible harmful side effects.\",\n"
            "                \"preconditions\": [\n"
            "                    \"(3) Cause(Hormone, Pain)\",\n"
            "                    \"(4) Cause(Hormone, Swelling)\",\n"
            "                    \"(7) Inhibit(Protein, Hormone)\",\n"
            "                    \"(8) Contain(ArthritisMedication, Protein)\"\n"
            "                ],\n"
            "                \"deduction_target\": \"Benefit(ArthritisMedication) > HarmfulSideEffect(ArthritisMedication)\",\n"
            "                \"deduction_steps\": [\n"
            "                    {\n"
            "                        \"step\": \"1\",\n"
            "                        \"task\": \"Identify the benefits of the medication based on inhibiting the hormone.\",\n"
            "                        \"expression\": \"Inhibit(Protein, Hormone) → Reduce(Pain) ∧ Reduce(Swelling)\",\n"
            "                        \"result\": \"Supported by context: The protein inhibits the hormone, which causes pain and swelling.\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"2\",\n"
            "                        \"task\": \"Determine if the context provides information about harmful side effects.\",\n"
            "                        \"expression\": \"HarmfulSideEffect(ArthritisMedication)\",\n"
            "                        \"result\": \"Not supported. The context does not mention any side effects of the medication.\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"3\",\n"
            "                        \"task\": \"Derivation fails.\",\n"
            "                        \"expression\": \"Derivation cannot proceed.\",\n"
            "                        \"result\": \"Cannot compare benefits and harmful side effects due to lack of information on side effects.\"\n"
            "                    }\n"
            "                ],\n"
            "                \"is_correct\": false\n"
            "            }\n"
            "        ]\n"
            "    }\n"
            "}\n\n"
            "---\n\n"
            "### Tips for Option Analysis:\n"
            "- **Preconditions**: Only include logical expressions that are directly relevant to the option being analyzed. Avoid listing all possible preconditions.\n"
            "- **Deduction Steps**: Ensure each step logically follows from the previous one based on the preconditions. If a step cannot be completed due to insufficient support from the preconditions, indicate the failure and stop further deductions for that option.\n"
            "- **is_correct**: This field should be true only for the option that matches the answer index. All other options should be false.\n"
            "- **Format Consistency**: Maintain the same JSON structure and field naming conventions across all options to ensure uniformity and ease of data extraction.\n"
            "- **Logical Accuracy**: Ensure that all logical expressions and deductions accurately reflect the relationships defined by the predicates and constants.\n\n"
            "### Your Task:\n"
            "Analyze the following Input data and generate the option_analysis section as per the example above. Replace the xxx placeholders in the example with actual data derived from the input.\n\n"
            "### Input Data:\n\n"
            "{{input_data_here}}\n"
            "### Please generate the option_analysis section based on the above input data."
        )
    }
]

messages_BQA = [
    {
        "role": "system",
        "content": (
            "Instructions: Please analyze the following binary question data. Since BQA is treated as an MCQA with a single option derived from the question, perform the analysis for this single option. Extract the relevant preconditions, define the deduction target, outline the deduction steps based on the provided Predicates, Constants, and Logical Expressions, and determine whether the option is correct based on the given answer index. Follow these specific rules:\n"
            "1. **Output** should be a JSON object containing only the `option_analysis` field, which is a list of analyses for each option.\n"
            "2. For the option_analysis, include the following fields:\n"
            "   - option_index: The index of the option (0-based).\n"
            "   - option_text: The text of the option (extracted from the question).\n"
            "   - preconditions: A list of relevant preconditions from the Logical Expressions that pertain to the option.\n"
            "   - deduction_target: The abstracted logical conclusion that the option is attempting to establish.\n"
            "   - deduction_steps: A step-by-step logical deduction process from the preconditions to the deduction target. Each step should include:\n"
            "     - step: The step number.\n"
            "     - task: A description of what is being checked or inferred in this step.\n"
            "     - expression: The logical expression used in this step, enclosed in backticks.\n"
            "     - result: The outcome of this step, enclosed in backticks.\n"
            "     - If a deduction step cannot proceed due to unsupported premises, indicate the failure and terminate further steps.\n"
            "   - is_correct: A boolean indicating whether the option is correct (true) or not (false). This should align with the answer field. When 'answer' is equal to 'yes', the value is true, otherwise the value is false.\n"
            "3. **Format Requirements**:\n"
            "   - The output must strictly follow the JSON structure as shown in the example.\n"
            "   - Ensure consistency in field naming and hierarchy.\n"
            "   - Do not include any additional fields or information not specified in the example.\n"
            "4. **Important Considerations**:\n"
            "   - Only include preconditions that are directly relevant to the option being analyzed.\n"
            "   - Maintain logical rigor in deduction steps, ensuring each step follows from the previous ones based on the preconditions.\n"
            "   - Avoid including unrelated preconditions to minimize complexity and enhance clarity.\n"
        )
    },
    {
        "role": "user",
        "content": (
            "{\n"
            "    \"Example\": {\n"
            "        \"context\": \"All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.\",\n"
            "        \"question\": \"Rina is a person who jokes about being addicted to caffeine or unaware that caffeine is a drug.\",\n"
            "        \"options\": [\n"
            "            \"Rina is a person who jokes about being addicted to caffeine or unaware that caffeine is a drug.\"\n"
            "        ],\n"
            "        \"answer\": \"yes\",\n"
            "        \"Predicates\": [\n"
            "            \"RegularlyDrink(x, y): Represents 'x' regularly drinking 'y' (e.g., a person regularly drinking coffee).\",\n"
            "            \"DependentOn(x, y): Represents 'x' being dependent on 'y' (e.g., a person dependent on caffeine).\",\n"
            "            \"JokeAbout(x, y): Represents 'x' joking about 'y' (e.g., a person joking about being addicted to caffeine).\",\n"
            "            \"UnawareThat(x, y): Represents 'x' being unaware that 'y' (e.g., a person unaware that caffeine is a drug).\",\n"
            "            \"IsStudent(x): Represents 'x' being a student.\",\n"
            "            \"IsNeither(x, y): Represents 'x' being neither 'y' (used for expressing negation of multiple conditions).\"\n"
            "        ],\n"
            "        \"Constants\": [\n"
            "            \"People: Generic individuals.\",\n"
            "            \"Coffee: The beverage being consumed.\",\n"
            "            \"Caffeine: The substance people can be dependent on.\",\n"
            "            \"Rina: A specific person mentioned in the context.\"\n"
            "        ],\n"
            "        \"Logical Expressions\": [\n"
            "            \"DependentOn(People, Caffeine) ← RegularlyDrink(People, Coffee)\",\n"
            "            \"JokeAbout(People, Caffeine) ∨ RegularlyDrink(People, Coffee)\",\n"
            "            \"UnawareThat(People, Caffeine) ← JokeAbout(People, Caffeine)\",\n"
            "            \"IsStudent(Rina) ∧ UnawareThat(Rina, Caffeine) ∨ ¬IsStudent(Rina) ∧ ¬UnawareThat(Rina, Caffeine)\",\n"
            "            \"¬DependentOn(Rina, Caffeine) ∧ IsStudent(Rina) → (DependentOn(Rina, Caffeine) ∧ IsStudent(Rina)) ∨ ¬(DependentOn(Rina, Caffeine) ∧ IsStudent(Rina))\"\n"
            "        ],\n"
            "    },\n\n"
            "    \"Output\": {\n"
            "        \"option_analysis\": [\n"
            "            {\n"
            "                \"option_index\": 0,\n"
            "                \"option_text\": \"Rina is a person who jokes about being addicted to caffeine or unaware that caffeine is a drug.\",\n"
            "                \"preconditions\": [\n"
            "                    \"JokeAbout(People, Caffeine) ∨ RegularlyDrink(People, Coffee)\",\n"
            "                    \"UnawareThat(People, Caffeine) ← JokeAbout(People, Caffeine)\"\n"
            "                ],\n"
            "                \"deduction_target\": \"JokeAbout(Rina, Caffeine) ∨ UnawareThat(Rina, Caffeine)\",\n"
            "                \"deduction_steps\": [\n"
            "                    {\n"
            "                        \"step\": \"1\",\n"
            "                        \"task\": \"Instantiate the general disjunction for Rina from the population-level statement.\",\n"
            "                        \"expression\": \"JokeAbout(Rina, Caffeine) ∨ RegularlyDrink(Rina, Coffee)\",\n"
            "                        \"result\": \"Derived from JokeAbout(People, Caffeine) ∨ RegularlyDrink(People, Coffee)\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"2\",\n"
            "                        \"task\": \"Apply the implication that joking about caffeine leads to being unaware of it for Rina.\",\n"
            "                        \"expression\": \"UnawareThat(Rina, Caffeine) ← JokeAbout(Rina, Caffeine)\",\n"
            "                        \"result\": \"If Rina jokes about caffeine, then Rina is unaware that caffeine is a drug.\"\n"
            "                    },\n"
            "                    {\n"
            "                        \"step\": \"3\",\n"
            "                        \"task\": \"Combine the instantiated disjunction with the implication to derive the final conclusion.\",\n"
            "                        \"expression\": \"JokeAbout(Rina, Caffeine) ∨ UnawareThat(Rina, Caffeine)\",\n"
            "                        \"result\": \"Since JokeAbout(Rina, Caffeine) implies UnawareThat(Rina, Caffeine), the disjunction holds.\"\n"
            "                    }\n"
            "                ],\n"
            "                \"is_correct\": true\n"
            "            }\n"
            "        ]\n"
            "    }\n"
            "}\n\n"
            "---\n\n"
            "### Tips for Option Analysis:\n"
            "- **Preconditions**: Only include logical expressions that are directly relevant to the option being analyzed. Avoid listing all possible preconditions.\n"
            "- **Deduction Steps**: Ensure each step logically follows from the previous one based on the preconditions. If a step cannot be completed due to insufficient support from the preconditions, indicate the failure and stop further deductions for that option.\n"
            "- **is_correct**: This field should be true only for the option that matches the answer field. Since BQA has only one option, is_correct should align with the answer field.\n"
            "- **Format Consistency**: Maintain the same JSON structure and field naming conventions across all options to ensure uniformity and ease of data extraction.\n"
            "- **Logical Accuracy**: Ensure that all logical expressions and deductions accurately reflect the relationships defined by the predicates and constants.\n\n"
            "### Your Task:\n"
            "Analyze the following Input data and generate the option_analysis section as per the example above. Replace the xxx placeholders in the example with actual data derived from the input.\n\n"
            "### Input Data:\n\n"
            "{{input_data_here}}\n"
            "### Please generate the option_analysis section based on the above input data."
        )
    }
]

def format_messages(example):
    """
    根据 example['task type']，选择对应的消息模板，并将
    context, question, options, answer 等插入模板中，返回带 prompt 的 example。
    """

    task_type = example.get('task type', '')

    context = example.get('context', '')
    question = example.get('question', '')
    options = example.get('options', [])
    answer = example.get('answer', '')
    predicates = example.get('Predicates', [])
    constants = example.get('Constants', [])
    logical_expressions = example.get('Logical Expressions', [])

    # 根据不同的 task type 选择不同模板
    if task_type == "MCQA":
        selected_messages = copy.deepcopy(messages_MCQA)
        # 构造一个要插入的 Input Data 字符串
        input_data_str = (
            "{\n"
            f"\"Context\": \"{context}\",\n"
            f"\"Question\": \"{question}\",\n"
            f"\"Options\": {json.dumps(options, ensure_ascii=False)},\n"
            f"\"Answer\": \"{answer}\",\n"
            f"\"Predicates\": {json.dumps(predicates, ensure_ascii=False)},\n"
            f"\"Constants\": {json.dumps(constants, ensure_ascii=False)},\n"
            f"\"Logical Expressions\": {json.dumps(logical_expressions, ensure_ascii=False)}\n"
            "}"
        )
        # 将 input_data_str 注入到模板 messages_MCQA[1]['content'] 的占位符 {{input_data_here}}
        user_content = selected_messages[1]['content'].replace("{{input_data_here}}", input_data_str)
        selected_messages[1]['content'] = user_content

    elif task_type == "BQA":
        selected_messages = copy.deepcopy(messages_BQA)
        # 对于 BQA，将 question 作为唯一的 option
        input_data_str = (
            "{\n"
            f"\"Context\": \"{context}\",\n"
            f"\"Question\": \"{question}\",\n"
            f"\"Options\": \"{question}\",\n"  # BQA只有一个选项，所以直接放 question
            f"\"Answer\": \"{answer}\",\n"
            f"\"Predicates\": {json.dumps(predicates, ensure_ascii=False)},\n"
            f"\"Constants\": {json.dumps(constants, ensure_ascii=False)},\n"
            f"\"Logical Expressions\": {json.dumps(logical_expressions, ensure_ascii=False)}\n"
            "}"
        )
        user_content = selected_messages[1]['content'].replace("{{input_data_here}}", input_data_str)
        selected_messages[1]['content'] = user_content

    else:
        print(f"Example ID {example['id']}: Unknown task type '{task_type}'. Skipping.")
        return {'prompt': ''}

    # 将所有消息串联成一个 prompt
    prompt = ""
    for msg in selected_messages:
        prompt += f"{msg['role']}: {msg['content']}\n"

    example['prompt'] = prompt
    return example



def parse_output(example, output):
    """
    将 pipeline 的输出（list 或 dict）解析成 JSON，并存入 example['option_analysis']
    """
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
        return example

    # 尝试找到 'option_analysis' 字段的位置
    option_analysis_start = assistant_content.find('"option_analysis"')
    if option_analysis_start == -1:
        print(f"Example ID {example['id']}: 'option_analysis' field not found in assistant content.")
        example['assistant_content'] = assistant_content
        return example

    # 从 'option_analysis' 开始提取
    json_start = assistant_content.find('{', option_analysis_start)
    if json_start == -1:
        print(f"Example ID {example['id']}: '{option_analysis_start}' does not contain '{'{'}'.")
        example['assistant_content'] = assistant_content
        return example

    # 试图找到 JSON 对象的结尾
    # 使用简单的括号匹配算法
    brace_count = 0
    json_end = json_start
    for i, char in enumerate(assistant_content[json_start:], start=json_start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1  # 包含 '}'
                break

    json_str = assistant_content[json_start:json_end]

    # 尝试将提取出的字符串解析为 JSON
    try:
        response_json = json.loads(json_str)
        option_analysis = response_json.get('option_analysis', [])
        example['option_analysis'] = option_analysis
    except json.JSONDecodeError as e:
        print(f"Example ID {example['id']}: Failed to parse JSON from assistant content: {e}")
        print(f"Extracted JSON snippet: {json_str[:500]}")  # 打印前 500 个字符
        example['assistant_content'] = assistant_content

    # 移除 prompt 字段（可选）
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

    # 根据需要过滤 split
    # chunk_dataset = chunk_dataset.filter(lambda x: x['split'] == 'train')

    # 过滤掉 ID 小于 start_id 的示例
    chunk_dataset = chunk_dataset.filter(lambda x: x['id'] >= start_id)

    # 对数据块进行预处理（生成 prompt）
    chunk_dataset = chunk_dataset.map(format_messages)

    # 过滤掉 prompt 为空的示例（task type 未识别时）
    chunk_dataset = chunk_dataset.filter(lambda x: x.get('prompt', '') != '')

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
                    continue
        print(f"Found {len(processed_ids)} existing examples in {output_file}")

    # 获取待处理的示例
    examples_to_process = [ex for ex in chunk_dataset if ex['id'] not in processed_ids]

    if not examples_to_process:
        print(f"No new examples to process in chunk {chunk_idx}.")
        start_id = 0  # 重置 start_id
        continue

    # 定义批处理大小
    batch_size = 128  # 每处理 128 条数据保存一次结果
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
                batch_size=16  # 可根据显存大小进行调整
            )
        except Exception as e:
            print(f"Error during generation in chunk {chunk_idx}, batch {batch_idx}: {e}")
            continue

        # 处理输出结果并逐条保存
        with open(output_file, 'a', encoding='utf-8') as f:
            for ex, output in zip(batch_examples, outputs):
                processed_example = parse_output(ex, output)
                json_line = json.dumps(processed_example, ensure_ascii=False)
                f.write(json_line + '\n')

        print(f"Saved batch {batch_idx + 1}/{num_batches} of chunk {chunk_idx} to {output_file}")

    print(f"Chunk {chunk_idx} processed and saved to {output_file}")

    # 重置 start_id，为下一个数据块准备
    start_id = 0

print("All data processing completed.")
